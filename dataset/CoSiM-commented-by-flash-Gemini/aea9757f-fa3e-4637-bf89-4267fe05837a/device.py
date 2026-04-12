"""
@file device.py
@brief Distributed sensor unit simulation with static task partitioning and cyclic synchronization.
@details Implements a peer-to-peer architecture where sensing units perform synchronized 
data aggregation. Utilizes a class-level shared barrier and lock registry, with tasks 
distributed across a fixed pool of worker threads.
"""

from threading import Lock, Semaphore, Event, Thread
from sets import Set

class Barrier(object):
    """
    @brief implementation of a two-phase cyclic barrier using semaphores.
    Functional Utility: Provides recurring synchronization points for a group of threads.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # State: Thread counters wrapped in lists for mutable reference in phase logic.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Executes the two-turnstile protocol.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Turnstile logic: Holds threads until all participants arrive, then releases them.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Wakes all threads blocked on this turnstile.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for next cycle.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    @brief Controller for a sensing entity in a distributed network.
    Functional Utility: Manages local data, coordinates parallel task execution, and 
    shares synchronization resources via static class attributes.
    """

    # Static Synchronization Resources: Shared by all Device instances.
    barrier = None
    lock_list = []

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Dictionary of local sensor readings.
        @param supervisor entity providing topology discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.scripts = []
        # Lifecycle: Spawns the main device management thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.devices = []

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of shared cluster resources.
        Logic: Designates the first calling device as the allocator for the 
        global barrier and location-specific locks.
        """
        self.devices = devices
        if Device.barrier is None:
            Device.barrier = Barrier(len(devices))
        
        if not Device.lock_list:
            # Discovery: Collects all unique sensor zones/locations across the cluster.
            zones = []
            for dev in devices:
                zones.extend(dev.sensor_data.keys())
            # Allocation: One mutual exclusion lock per unique location.
            Device.lock_list = [Lock() for _ in range(len(Set(zones)))]

    def assign_script(self, script, location):
        """
        @brief Schedules a processing script for the current unit of time.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Protocol: Signals end of the script submission phase.
            self.scripts_received.set()

    def get_data(self, location):
        """
        @brief retrieval of local sensor state.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        @brief Update of local sensor state.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device lifecycle thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Orchestration thread managing timepoint cycles and thread distribution.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main loop: discovery -> partitioning -> parallel execution -> sync.
        """
        while True:
            # Discovery: Fetches current network topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for script assignments.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            /**
             * Block Logic: Task partitioning.
             * Logic: Distributes scripts into 8 worker buckets using round-robin 
             * assignment to ensure load balancing.
             */
            tasks = [[] for _ in xrange(8)]
            for i, task in enumerate(self.device.scripts):
                tasks[i % 8].append(task)

            # Execution: Spawns one thread per non-empty partition.
            script_threads = []
            for i in xrange(8):
                if tasks[i]:
                    thr = ScriptThread(self.device, neighbours, tasks[i])
                    script_threads.append(thr)
                    thr.start()

            # Sync: Blocks until all local worker threads finish processing.
            for thread in script_threads:
                thread.join()

            # Global Sync: Ensure all devices in the network align at the barrier.
            Device.barrier.wait()

class ScriptThread(Thread):
    """
    @brief Worker thread that executes a sequence of data aggregation tasks.
    Architecture: Implements distributed Map-Reduce for each assigned script.
    """
    
    def __init__(self, device, neighbours, scripts):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.scripts = scripts

    def run(self):
        """
        @brief Execution logic for aggregation.
        Critical Section: Uses the global location-specific lock to ensure cluster-wide atomicity.
        """
        for (script, location) in self.scripts:
            # Lock: Exclusive access to this sensor's state across the peer network.
            Device.lock_list[location].acquire()
            
            script_data = []
            # Map Phase: Aggregates readings from neighbors.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Inclusion: local state.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Computation Phase.
                result = script.run(script_data)
                
                # Reduce/Update Phase: writes back the new state to all participants.
                for device in self.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            
            Device.lock_list[location].release()
