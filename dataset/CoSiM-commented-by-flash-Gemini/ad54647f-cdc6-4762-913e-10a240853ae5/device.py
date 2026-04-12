"""
@file device.py
@brief Distributed sensing and data processing system with cyclic synchronization.
@details Implements a peer-to-peer network of devices that execute aggregation 
scripts in parallel. Coordinates global state using a custom two-phase 
semaphore-based barrier and location-specific mutual exclusion locks.
"""

from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    @brief Implementation of a two-turnstile cyclic barrier using semaphores.
    Functional Utility: Synchronizes a fixed group of threads across recurring cycles.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # State: Thread counters wrapped in lists to allow pass-by-reference modification.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief executes the two-phase barrier protocol.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Turnstile logic: Blocks threads until the threshold is reached, then releases them.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Release the current turnstile.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for the next cycle.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    @brief Controller for an autonomous sensing unit.
    Functional Utility: Manages local data buffers, stages aggregation scripts, and 
    shares synchronization primitives with network peers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Dictionary of local sensor readings.
        @param supervisor entity providing neighborhood discovery services.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()
        # Lifecycle: Spawns the main management thread.
        self.thread = DeviceThread(self)
        self.barrier = None
        self.list_thread = []
        # Pre-allocation: Registry for up to 100 location-specific locks.
        self.location_lock = [None] * 100
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Cluster-wide initialization of shared synchronization resources.
        Logic: All devices in the provided list adopt a single shared barrier instance.
        """
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        @brief Schedules a processing task for the current unit of time.
        Logic: Ensures a shared lock exists for the target location across the cluster.
        """
        flag = 0
        if script is not None:
            self.scripts.append((script, location))
            
            # Lock Discovery: Checks if a peer has already allocated a lock for this location.
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        flag = 1
                        break

                # Allocation: Creates a new lock if no peer has one.
                if flag == 0:
                    self.location_lock[location] = Lock()
            
            self.script_received.set()
        else:
            # Protocol: Signals end of the script submission phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieval of local sensor readings.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        @brief Update of local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device lifecycle thread.
        """
        self.thread.join()


class MyThread(Thread):
    """
    @brief Worker thread for executing a single data aggregation script.
    Functional Utility: Implements a distributed Map-Reduce operation.
    """
    
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief Execution logic for aggregation.
        Critical Section: Uses the location-specific shared lock to ensure atomicity.
        """
        self.device.location_lock[self.location].acquire()
        script_data = []
        
        # Map Phase: Collects readings from neighborhood peers.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        # Inclusion: Local device data.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Computation: Executes the aggregation logic.
            result = self.script.run(script_data)
            
            # Reduce/Update Phase: Propagates result to all participants.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
            
        self.device.location_lock[self.location].release()


class DeviceThread(Thread):
    """
    @brief Management thread coordinating discrete timepoint cycles.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main orchestration loop: discovery -> execution -> synchronization.
        """
        while True:
            # Discovery: Queries the network supervisor for current neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until task assignment phase concludes.
            self.device.timepoint_done.wait()

            /**
             * Block Logic: Parallel script execution.
             * Logic: Spawns one worker thread per (script, location) pair and joins them all.
             */
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            for thread_elem in self.device.list_thread:
                thread_elem.start()
            
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            
            # Clean up thread registry for next cycle.
            self.device.list_thread = []

            # Reset unit of time state and sync at global barrier.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
