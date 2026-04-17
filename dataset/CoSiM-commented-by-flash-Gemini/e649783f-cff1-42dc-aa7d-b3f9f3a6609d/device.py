"""
@e649783f-cff1-42dc-aa7d-b3f9f3a6609d/device.py
@brief Distributed sensor processing simulation using batch task execution and shared global locking.
* Algorithm: Phased worker dispatching (8 threads per batch) with on-demand shared lock discovery and two-phase semaphore barriers.
* Functional Utility: Orchestrates simulation timepoints by dividing local scripts into manageable batches, coordinating neighbor data aggregation and synchronized updates across the cluster.
"""

from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier(object):
    """
    @brief Two-phase synchronization barrier implementation using counting semaphores.
    * Algorithm: Dual-stage arrival/release logic to prevent thread overruns between consecutive simulation steps.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the barrier state with target count and phase primitives.
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
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings and coordination infrastructure.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the main coordinator thread.
        """
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.threads = []
        self.devices = []
        self.semafor = Semaphore(0)
        self.timepoint_done = Event() # Intent: Signals completion of task delivery for the current phase.
        self.thread = SupervisorThread(self)
        self.thread.start()
        self.num_scr = 8 # Domain: Concurrency Scaling - max batch size for parallel slaves.
        self.lock = [None] * 100 # Intent: Registry for shared locks indexed by sensor location ID.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization setup and node registration.
        Invariant: Root node initializes the shared barrier and propagates its local lock registry.
        """
        if self.barrier is None:
            self.barrier = ReusableBarrier(len(devices))
            for dev in devices:
                dev.lock = self.lock
                if dev.barrier is None:
                    dev.barrier = self.barrier

        for dev in devices:
            if dev is not None:
                self.devices.append(dev)

    def assign_script(self, script, location):
        """
        @brief Receives a processing task and handles on-demand shared lock discovery.
        Logic: Attempts to find existing lock for the location from peers before creating a new one.
        """
        if script is not None:
            self.scripts.append((script, location))
            if self.lock[location] is None:
                # Logic: Scan neighbor nodes for existing lock instance to ensure cluster-wide mutual exclusion.
                for device in self.devices:
                    if device.lock[location] is not None:
                        self.lock[location] = device.lock[location]
                        break
                    self.lock[location] = Lock()
            self.script_received.set()
        else:
            # Logic: Signals that script delivery for this timepoint is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Standard data retrieval interface for local sensor readings.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Standard data update interface for local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Terminates the device coordination thread.
        """
        self.thread.join()


class SupervisorThread(Thread):
    """
    @brief Main management thread coordinating batch-based task dispatching.
    """
    
    def __init__(self, device):
        """
        @brief Initializes the supervisor for a specific device node.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core lifecycle loop of the device node coordination.
        Algorithm: Iterative batch execution (8 Slaves at a time) followed by cluster barrier alignment.
        """
        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighb = self.device.supervisor.get_neighbours()
            if neighb is None:
                break
            
            # Block Logic: Ensures all tasks for the current cycle have arrived.
            self.device.timepoint_done.wait()
            i = 0
            while i < len(self.device.scripts):
                # Batch Dispatch Phase.
                # Strategy: Spawns up to num_scr parallel slaves and waits for the entire batch to complete.
                for _ in range(0, self.device.num_scr):
                    pair = self.device.scripts[i]
                    new_thread = Slave(self.device, pair[1], neighb, pair[0])
                    self.device.threads.append(new_thread)
                    new_thread.start()
                    i = i + 1
                    if i >= len(self.device.scripts):
                        break
                
                # Internal Synchronization: Wait for all local batch workers to finish.
                for thread in self.device.threads:
                    thread.join()

            self.device.threads = []
            
            # Post-condition: Reset phase state and align all devices across the cluster.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

class Slave(Thread):
    """
    @brief worker thread implementing the execution of a single sensor script unit.
    """

    def __init__(self, device, location, neighbours, script):
        """
        @brief Initializes worker with task parameters and neighborhood context.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def run(self):
        """
        @brief Main execution logic for a single script unit.
        Algorithm: Resource-locked execution with distributed data aggregation and propagation.
        """
        # Pre-condition: Acquire shared global location lock for atomic distributed update.
        self.device.lock[self.location].acquire()
        script_data = []
        
        # Distributed Aggregation Phase: Collect readings from neighbors and local node.
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
            
        # Post-condition: Release global location lock.
        self.device.lock[self.location].release()
