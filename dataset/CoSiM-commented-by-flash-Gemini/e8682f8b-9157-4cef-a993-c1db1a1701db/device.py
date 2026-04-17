"""
@e8682f8b-9157-4cef-a993-c1db1a1701db/device.py
@brief Distributed sensor network simulation with state-driven persistent worker pools.
This module implements a coordinated parallel processing framework where persistent 
worker threads (Worker) are managed via an event-driven handshake protocol. Each 
worker operates in multiple modes (Run, Timepoint_End, Shutdown) to synchronize 
with the node orchestrator (DeviceThread). Consistency is maintained through a 
combination of spatial locks and data-readiness events, ensuring that temporal 
consensus is reached via two-phase barriers.

Domain: Persistent Thread Handshaking, State-Driven Workers, Distributed Consistency.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Implements a double-gate mechanism with semaphores to ensure 
    perfect temporal alignment and prevents thread overtaking between simulation steps.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: Number of participants in the rendezvous.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Orchestrates the two-phase synchronization rendezvous."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement and semaphore release."""
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
    Coordinator entity for a network node.
    Functional Utility: Manages local data state, coordinates global synchronization 
    resource distribution, and supervises a pool of persistent execution workers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.max_threads = 8
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Signaling events for simulation step transitions.
        self.notification = Event()
        self.timepoint_done = Event()
        self.notification.clear()
        self.timepoint_done.clear()
        
        # Spatial Synchronization resources.
        self.update_locks = {}
        self.read_locations = {}
        self.external_barrier = None
        self.internal_barrier = ReusableBarrier(self.max_threads)
        
        # Initialize and activate the local worker pool.
        self.workers = self.setup_workers()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_workers(self):
        """Spawns persistent worker thread objects."""
        workers = []
        i = 0
        while i < self.max_threads:
            workers.append(Worker(self))
            i += 1
        return workers

    def start_workers(self):
        """Activates all worker threads in the pool."""
        for i in range(0, self.max_threads):
            self.workers[i].start()

    def stop_workers(self):
        """Gracefully joins all worker threads."""
        for i in range(0, self.max_threads):
            self.workers[i].join()

    def setup_devices(self, devices):
        """
        Global synchronization initialization.
        Logic: Coordinator node (ID 0) initializes and shares the global barrier.
        """
        if self.device_id == 0:
            self.external_barrier = ReusableBarrier(len(devices))
        else:
            # Discovery: Busy-wait (polling) for Node 0 to initialize the shared barrier.
            for device in devices:
                if device.device_id == 0:
                    while device.external_barrier is None:
                        pass
                    self.external_barrier = device.external_barrier
                    break

    def assign_script(self, script, location):
        """
        Registers a computational task.
        Logic: Initializes spatial locks and readiness signals on-demand.
        """
        self.notification.set()
        if script is not None:
            if location not in self.update_locks:
                self.update_locks[location] = Lock()
                # Readiness Signal: event indicating that data is safe to read.
                self.read_locations[location] = Event()
                self.read_locations[location].set()
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Safe data retrieval.
        Functional Utility: Blocks until the readiness event for the location is set.
        @return: Sensor value or None.
        """
        if location not in self.sensor_data:
            return None
        else:
            if location not in self.read_locations:
                self.update_locks[location] = Lock()
                self.read_locations[location] = Event()
                self.read_locations[location].set()
            
            # Wait for any active updates to complete.
            self.read_locations[location].wait()
            return self.sensor_data[location]

    def set_data(self, location, data):
        """
        Atomic data update.
        Logic: Clears the readiness event before update to block readers, 
        and sets it after to release them.
        """
        if location in self.sensor_data:
            self.update_locks[location].acquire()
            # Mark location as busy/invalid for readers.
            self.read_locations[location].clear()
            self.sensor_data[location] = data
            # Signal completion to waiting readers.
            self.read_locations[location].set()
            self.update_locks[location].release()

    def shutdown(self):
        """Gracefully terminates all management and worker threads."""
        self.stop_workers()
        self.thread.join()


class DeviceThread(Thread):
    """
    Main orchestration thread for the node.
    Functional Utility: Implements a task distributor that identifies idle 
    workers and manages simulation phase transitions.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def find_free_worker(self):
        """Heuristic: linear scan for an idle thread in the pool."""
        for i in range(0, self.device.max_threads):
            if self.device.workers[i].is_free:
                return i
        return -1

    def run(self):
        """
        Main simulation execution loop.
        Algorithm: Iterative sequence of topology refresh, parallel task 
        distribution, and global consensus.
        """
        self.device.start_workers()

        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            
            if neighbours is None:
                # Dispatch shutdown signal to all workers.
                for i in range(0, self.device.max_threads):
                    self.device.workers[i].update(None, None, None, "end")
                break

            # Wait for task arrival signal.
            if len(self.device.scripts) == 0:
                self.device.notification.wait()

            curr_scr = 0
            # Block Logic: Workload distribution phase.
            # Logic: Continues until all scripts are assigned and supervisor signals end-of-step.
            while (curr_scr < len(self.device.scripts)) or \
                  (self.device.timepoint_done.is_set() is False):
                worker_idx = self.find_free_worker()
                if (worker_idx >= 0) and (curr_scr < len(self.device.scripts)):
                    # Assign computational task to an idle worker.
                    (script, location) = self.device.scripts[curr_scr]
                    self.device.workers[worker_idx].update(location, script, neighbours, "run")
                    curr_scr += 1
                else:
                    # Busy-wait for worker availability.
                    continue

            # Signal phase completion to the pool.
            for i in range(0, self.device.max_threads):
                self.device.workers[i].update(None, None, None, "timepoint_end")
            self.device.timepoint_done.clear()
            self.device.notification.clear()
            
            # Global consensus rendezvous.
            self.device.external_barrier.wait()


class Worker(Thread):
    """
    Persistent worker thread implementation.
    Functional Utility: Implements a mode-based state machine and uses a 
    double-event handshake (init_start/exec_start) with the node orchestrator.
    """

    def __init__(self, device):
        Thread.__init__(self)
        self.device = device
        # Handshake events for coordination with the DeviceThread.
        self.init_start = Event()
        self.exec_start = Event()
        self.location = None
        self.script = None
        self.neighbours = None
        self.is_free = True
        self.mode = ""
        self.exec_start.clear()
        self.init_start.set()

    def update(self, location, script, neighbours, mode):
        """
        Handshake interface for the orchestrator to provide new tasks.
        @param mode: Defines the next operational phase (run/timepoint_end/end).
        """
        self.init_start.wait()
        self.init_start.clear()
        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.mode = mode
        self.is_free = False
        self.exec_start.set()

    def run(self):
        """
        Worker execution loop.
        Algorithm: Blocks until exec_start is signaled, then dispatches based on mode.
        """
        while True:
            self.exec_start.wait()
            self.exec_start.clear()
            
            if self.mode == "end":
                # Exit signal received.
                break
            elif self.mode == "timepoint_end":
                # Local group synchronization at end of simulation step.
                self.device.internal_barrier.wait()
                self.is_free = True
                self.init_start.set()
            else:
                # Mode: 'run' - Execute computational logic.
                script_data = []
                
                # Neighborhood aggregation.
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        script_data.append(data)
                
                # Include local state.
                data = self.device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Compute result and propagate to all nodes in the graph.
                    result = self.script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    self.device.set_data(self.location, result)
                
                # Signal readiness for next task assignment.
                self.is_free = True
                self.init_start.set()
