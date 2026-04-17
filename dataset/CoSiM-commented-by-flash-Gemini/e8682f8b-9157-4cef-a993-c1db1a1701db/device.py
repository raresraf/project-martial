"""
@e8682f8b-9157-4cef-a993-c1db1a1701db/device.py
@brief Distributed sensor processing simulation using a persistent worker pool and event-driven task scheduling.
* Algorithm: Dynamic task allocation to a pool of 8 persistent `Worker` threads using a state-based coordinator (Control -> Worker via modes) and multi-level barriers.
* Functional Utility: Orchestrates simulation timepoints across a network of devices by managing asynchronous task execution and synchronized data propagation using event-based read-write locks.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    @brief Implementation of a two-phase synchronization barrier using semaphores.
    * Algorithm: Dual-stage arrival/release pattern to ensure consistent thread alignment across repeated simulation cycles.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier state and its internal phase control primitives.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Intent: Shared mutable counter.
        self.count_threads2 = [self.num_threads] # Intent: Shared mutable counter.
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Synchronizes the calling thread through both stages of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single synchronization stage.
        Invariant: The last thread to arrive releases the entire group and resets the counter.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                i = 0
                while i < self.num_threads:
                    threads_sem.release()
                    i += 1
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    


class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, internal worker pool, and shared synchronization infrastructure.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and prepares its internal thread infrastructure.
        """
        self.max_threads = 8 # Domain: Concurrency Scaling - fixed workers per node.
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.notification = Event() # Intent: Signals the coordinator when the first script arrives.
        self.timepoint_done = Event() # Intent: Signals completion of task delivery for the current phase.
        self.notification.clear()
        self.timepoint_done.clear()
        
        self.update_locks = {}    # Intent: Maps locations to mutexes for serialized updates.
        self.read_locations = {}  # Intent: Maps locations to events acting as read barriers during updates.
        
        self.external_barrier = None # Intent: Global barrier for cluster-wide alignment.
        self.internal_barrier = ReusableBarrier(self.max_threads) # Intent: Internal worker coordination.
        
        # Logic: Bootstraps the persistent worker workforce.
        self.workers = self.setup_workers()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_workers(self):
        """
        @brief Instantiates the worker thread pool.
        """
        workers = []
        i = 0
        while i < self.max_threads:
            workers.append(Worker(self))
            i += 1
        return workers

    def start_workers(self):
        """
        @brief Signals all worker threads to begin their execution loops.
        """
        for i in range(0, self.max_threads):
            self.workers[i].start()

    def stop_workers(self):
        """
        @brief Gracefully terminates all workers in the pool.
        """
        for i in range(0, self.max_threads):
            self.workers[i].join()

    def setup_devices(self, devices):
        """
        @brief Global synchronization setup.
        Invariant: Root node (ID 0) initializes and propagates the cluster-wide barrier.
        """
        if self.device_id == 0:
            self.external_barrier = ReusableBarrier(len(devices))
        else:
            # Logic: Busy-wait discovery of the leader's barrier instance.
            for device in devices:
                if device.device_id == 0:
                    while device.external_barrier is None:
                        pass
                    self.external_barrier = device.external_barrier
                    break

    def assign_script(self, script, location):
        """
        @brief Top-level interface for script arrival.
        Logic: Registers necessary locks and read barriers for new sensor locations.
        """
        self.notification.set()
        if script is not None:
            if location not in self.update_locks:
                # Logic: Initializes event-based RW lock for the target location.
                self.update_locks[location] = Lock()
                self.read_locations[location] = Event()
                self.read_locations[location].set()
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Synchronized retrieval of sensor data.
        Algorithm: Read-barrier wait ensures that data is not fetched during an active update phase.
        """
        if location not in self.sensor_data:
            return None
        else:
            # Logic: Lazy initialization of read barriers for remote data access.
            if location not in self.read_locations:
                self.update_locks[location] = Lock()
                self.read_locations[location] = Event()
                self.read_locations[location].set()
            
            # Block Logic: Ensures atomic visibility of completed updates.
            self.read_locations[location].wait()
            return self.sensor_data[location]

    def set_data(self, location, data):
        """
        @brief Synchronized update of sensor data.
        Algorithm: Exclusive acquisition followed by read-barrier reset to block concurrent readers.
        """
        if location in self.sensor_data:
            self.update_locks[location].acquire()
            # Logic: Blocks subsequent readers.
            self.read_locations[location].clear()
            self.sensor_data[location] = data
            # Post-condition: Unblocks waiting readers.
            self.read_locations[location].set()
            self.update_locks[location].release()

    def shutdown(self):
        """
        @brief Terminates the device coordination thread and its workers.
        """
        self.stop_workers()
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Management thread coordinating task dispatching to the local worker pool.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator for a specific device node.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def find_free_worker(self):
        """
        @brief Strategy: Linear scan for an idle worker thread.
        """
        for i in range(0, self.device.max_threads):
            if self.device.workers[i].is_free:
                return i
        return -1

    def run(self):
        """
        @brief Core lifecycle loop of the coordinator.
        Algorithm: Mode-based worker orchestration (Run -> Timepoint End -> Global Barrier).
        """
        self.device.start_workers()

        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            if neighbours is None:
                # Logic: Shutdown signal - command workers to terminate.
                for i in range(0, self.device.max_threads):
                    self.device.workers[i].update(None, None, None, "end")
                break

            # Block Logic: Wait for the first task to arrive.
            if len(self.device.scripts) == 0:
                self.device.notification.wait()

            # Dispatch Phase: Distributes tasks to available workers as scripts arrive.
            curr_scr = 0
            while (curr_scr < len(self.device.scripts)) or \
                  (self.device.timepoint_done.is_set() is False):
                worker_idx = self.find_free_worker()
                if (worker_idx >= 0) and (curr_scr < len(self.device.scripts)):
                    # Logic: Updates worker mode to "run" and provides task parameters.
                    (script, location) = self.device.scripts[curr_scr]
                    self.device.workers[worker_idx].update(location, script, neighbours, "run")
                    curr_scr += 1
                else:
                    # Logic: Busy-wait polling for worker availability.
                    continue

            # Synchronization Phase 1: Local worker alignment.
            for i in range(0, self.device.max_threads):
                self.device.workers[i].update(None, None, None, "timepoint_end")
            
            # Post-condition: Reset phase state.
            self.device.timepoint_done.clear()
            self.device.notification.clear()
            
            # Synchronization Phase 2: Global cluster alignment.
            self.device.external_barrier.wait()


class Worker(Thread):
    """
    @brief Persistent worker thread implementing the execution of assigned sensor scripts.
    """

    def __init__(self, device):
        """
        @brief Initializes worker with its task-trigger events and local state.
        """
        Thread.__init__(self)
        self.device = device
        self.init_start = Event() # Intent: Triggers when worker is ready for new task parameters.
        self.exec_start = Event() # Intent: Triggers when task processing should begin.
        self.location = None
        self.script = None
        self.neighbours = None
        self.is_free = True
        self.mode = ""
        self.exec_start.clear()
        self.init_start.set()

    def update(self, location, script, neighbours, mode):
        """
        @brief Coordinator interface for updating worker state.
        Pre-condition: Must wait for worker to be in 'init' state.
        """
        self.init_start.wait()
        self.init_start.clear()
        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.mode = mode
        self.is_free = False
        # Logic: Signals worker to transition to execution phase.
        self.exec_start.set()

    def run(self):
        """
        @brief main loop for the persistent worker thread.
        Algorithm: Mode-driven execution machine (end, timepoint_end, run).
        """
        while True:
            # Block Logic: Waits for coordinator trigger.
            self.exec_start.wait()
            self.exec_start.clear()
            
            if self.mode == "end":
                # Logic: Shutdown path.
                break
            elif self.mode == "timepoint_end":
                # Logic: Phase alignment at the local barrier.
                self.device.internal_barrier.wait()
                self.is_free = True
                self.init_start.set()
            else:
                # Execution Phase.
                script_data = []
                
                # Distributed Aggregation: Collect readings from neighbors and self.
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
                
                if script_data != []:
                    # Execution and Propagation Phase.
                    result = self.script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    self.device.set_data(self.location, result)
                
                # Logic: Transitions back to idle state.
                self.is_free = True
                self.init_start.set()
