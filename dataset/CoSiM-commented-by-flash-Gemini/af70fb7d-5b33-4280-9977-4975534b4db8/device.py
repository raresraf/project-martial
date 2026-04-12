"""
@file device.py
@brief Distributed sensing and data processing system with cyclic synchronization and location-based locking.
@details Implements a peer-to-peer network of devices that execute aggregation scripts in 
parallel. Utilizes a custom two-phase semaphore-based barrier for cluster-wide temporal 
alignment and ensures data consistency through shared mutual exclusion locks per sensor location.
"""

from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    @brief Implementation of a two-turnstile cyclic barrier using semaphores.
    Functional Utility: Provides recurring synchronization points for a group of threads.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # State: Thread counters wrapped in lists for pass-by-reference mutable logic.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Executes the two-phase barrier protocol.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Turnstile logic: Blocks threads until all expected participants arrive.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Release all blocked threads for the current phase.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for reuse in the next cycle.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    @brief Logic controller for an autonomous sensing entity.
    Functional Utility: Manages local sensor data, organizes parallel task execution, and 
    shares synchronization resources with peers in the distributed network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of sensor location-reading pairs.
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
        # Pre-allocation: Shared registry for location-specific locks.
        self.location_lock = [None] * 100
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of shared cluster resources.
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
        @brief Schedules a processing script for the current unit of time.
        Logic: Ensures a shared Lock exists for the target location across the cluster.
        """
        flag = 0
        if script is not None:
            self.scripts.append((script, location))
            
            # Discovery: Checks if a peer device has already initialized a lock for this location.
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        flag = 1
                        break
                
                # Allocation: Creates a new lock if no peer has one yet.
                if flag == 0:
                    self.location_lock[location] = Lock()
            
            self.script_received.set()
        else:
            # Protocol: Signals end of the script submission phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief retrieval of local sensor readings.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        @brief update of local sensor readings.
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
    Architecture: Implements a distributed Map-Reduce operation.
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
        Critical Section: Uses location-specific locks to ensure atomicity across the cluster.
        """
        self.device.location_lock[self.location].acquire()
        script_data = []
        
        # Map Phase: Aggregates state from topological neighborhood.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        # Inclusion: local state.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Computation: executes aggregation logic.
            result = self.script.run(script_data)
            
            # Reduce/Update Phase: writes back results to all participants.
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
            # Discovery: Queries network supervisor for current neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Sync: Wait for task assignment phase to conclude.
            self.device.timepoint_done.wait()

            /**
             * Block Logic: Parallel script execution.
             * Logic: Spawns one worker thread per (script, location) pair and joins them.
             */
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            for thread_elem in self.device.list_thread:
                thread_elem.start()
            
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            
            # Clean up thread list for reuse in next cycle.
            self.device.list_thread = []

            # Global Sync: Ensure all cluster members align at the barrier.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
