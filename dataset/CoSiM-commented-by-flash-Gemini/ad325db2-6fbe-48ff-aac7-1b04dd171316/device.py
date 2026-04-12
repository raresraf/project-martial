"""
@file device.py
@brief Distributed sensor processing system with batched parallel execution and cyclic synchronization.
@details Implements a network of sensing units that perform synchronized data aggregation. 
Uses a tiered threading model (Master and Mini-workers) and a custom semaphore-based 
barrier for coordinating temporal timepoints across the cluster.
"""

from threading import Thread, Semaphore, Event, Lock

class ReusableBarrierSem(object):
    """
    @brief Two-phase cyclic barrier implementation using semaphores.
    Functional Utility: Provides reliable synchronization points for multi-threaded cycles.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Barrier protocol: Phase 1 (Collect) followed by Phase 2 (Release/Reset).
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief First turnstile: Ensures all threads reach the barrier before any proceed.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Wakes all threads blocked on phase 1.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second turnstile: Prevents threads from cycling back to phase 1 too early.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    @brief Controller for an autonomous sensing unit in a distributed cluster.
    Functional Utility: Manages local data, task staging, and resource sharing 
    (barriers/locks) across the peer network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Dictionary of local sensor location-reading pairs.
        @param supervisor Coordination entity for topology management.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Lifecycle: Spawns the device orchestrator thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.locks = []
        # Metric: Maximum sensor location ID to determine lock count.
        self.nrlocks = max(sensor_data) if sensor_data else 0

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of cluster-wide synchronization primitives.
        Logic: Designates device 0 as the master allocator for barriers and locks.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            for device in devices:
                device.barrier = self.barrier
        
            # Dynamic Lock Allocation: Creates a lock for every possible sensor location.
            listmaxim = [d.nrlocks for d in devices]
            number = max(listmaxim)
            self.locks = [Lock() for _ in range(number + 1)]
            for device in devices:
                device.locks = self.locks

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing script for the current unit of time.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Protocol: Signals end of the script assignment phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieval of local sensor readings.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        @brief Updates local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device lifecycle thread.
        """
        self.thread.join()

class MiniDeviceThread(Thread):
    """
    @brief Worker thread for executing a single data aggregation script.
    Functional Utility: Implements a distributed Map-Reduce operation.
    """
    
    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        @brief Execution logic for aggregation.
        Critical Section: Uses the shared location lock to ensure atomic updates.
        """
        self.device.locks[self.location].acquire()
        script_data = []
        
        # Map Phase: Aggregates state from topological neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Inclusion: Self state.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Computation: Executes aggregation logic.
            result = self.script.run(script_data)
            
            # Reduce/Update Phase: Disseminates result to all participants.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        
        self.device.locks[self.location].release()


class DeviceThread(Thread):
    """
    @brief Orchestration thread managing timepoint cycles and task batching.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main loop: discovery -> batched execution -> synchronization.
        """
        while True:
            # Topology Discovery: Fetches current neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Sync: Wait for script assignment phase to conclude.
            self.device.timepoint_done.wait()
            
            /**
             * Block Logic: Batched parallel script execution.
             * Optimization: Processes scripts in chunks of 8 to balance parallelism vs overhead.
             */
            num_scripts = len(self.device.scripts)
            for i in range(0, num_scripts, 8):
                batch_threads = []
                # Functional Utility: Spawns workers for the current batch.
                for j in range(i, min(i + 8, num_scripts)):
                    script, location = self.device.scripts[j]
                    thread = MiniDeviceThread(self.device, script, location, neighbours)
                    batch_threads.append(thread)
                    thread.start()
                
                # Sync: Waits for the current batch to finalize before starting the next.
                for thread in batch_threads:
                    thread.join()
            
            # Global Sync: Ensure all cluster members align at the end of the unit of time.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
