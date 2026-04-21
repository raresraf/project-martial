
"""
@file device.py
@brief Distributed device simulation framework with multi-threaded script execution.

Functional Intent: Implements a parallel execution environment for autonomous 
devices in a simulated network. Features a hierarchical coordination model where 
each device utilizes a dedicated management thread to orchestrate parallel 
worker threads (SingleDeviceThread). Coordinates neighborhood discovery via a 
supervisor, ensures eventual consistency through distributed location-based 
locking, and synchronizes time-discrete epochs via a custom reusable semaphore barrier.

Domain: Production Systems, Distributed Computing, Parallel Processing.
"""

from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """
    @brief Represents a computational entity in the simulated distributed swarm.
    
    Functional Utility: Manages local sensor data and coordinates task processing. 
    Collaborates with peers to establish shared synchronization primitives (barriers 
    and lock pools) and manages the lifecycle of local management and worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device and préparer its orchestration state.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Synchronization: Events for task delivery and epoch boundary signaling.
        self.script_received = Event()
        self.timepoint_done = Event()
        
        # Logic: Dedicated management thread for high-level orchestration.
        self.thread = DeviceThread(self)
        self.thread.start()
        
        self.barrier = None
        self.map_locations = None # Logic: Shared lock pool for location-specific data protection.


    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Orchestrates the cluster-wide setup of shared synchronization resources.
        
        Algorithm: Single-node master initialization (Lowest ID device).
        Logic: Elects a master node to generate a global barrier and a distributed 
        lock map, ensuring all peers utilize identical synchronization objects.
        """
        
        flag = True
        device_number = len(devices)

        # Block Logic: Master node election.
        for dev in devices:
            if self.device_id > dev.device_id:
                flag = False

        if flag == True:
            # Block Logic: Resource bootstrapping.
            # Invariant: Initializes exactly one global barrier and a unified lock map.
            barrier = ReusableBarrierSem(device_number)
            map_locations = {}
            for dev in devices:
                dev.barrier = barrier
                
                # Logic: Discovers all unique data locations across the swarm and assigns mutexes.
                new_locs = list(set(dev.sensor_data) - set(map_locations))
                for i in new_locs:
                    map_locations[i] = Lock()
                    
                dev.map_locations = map_locations

    def assign_script(self, script, location):
        """
        @brief Binds a transformation script to a specific data location for the current step.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: Signaling None indicates the end of task assignment for the current epoch.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of sensor data. (Requires external locking).
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates local sensor state.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the orchestration thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Management thread responsible for task distribution and worker lifecycle.
    
    Logic: Orchestrates the transition between topology discovery, local parallel 
    execution, and global rendezvous points.
    """

    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        @brief Core operational cycle for the device manager.
        
        Algorithm: Discrete time-step orchestration.
        Logic: 
        1. Identifies neighbors via supervisor.
        2. Waits for all epoch-specific scripts to be delivered.
        3. Spawns a pool of worker threads to process the assigned scripts in parallel.
        4. Synchronizes at the global barrier after all local workers finish.
        """
        while True:
            self.device.timepoint_done.clear()
            neighbours = self.device.supervisor.get_neighbours()
            
            # Pre-condition: Shutdown signal from supervisor.
            if neighbours is None:
                break
                
            self.device.timepoint_done.wait()
            
            script_list = []
            thread_list = []
            for script in self.device.scripts:
                script_list.append(script)
            
            # Optimization: Fixed worker pool size (8 threads) to handle local script batch.
            for i in xrange(8):
                thread = SingleDeviceThread(self.device, script_list, neighbours, 0)
                thread.start()
                thread_list.append(thread)
            
            # Synchronization: Blocks manager until all local scripts for this epoch are processed.
            for i in xrange(len(thread_list)):
                thread_list[i].join()
            
            # Synchronization: Global rendezvous for step alignment across the network.
            self.device.barrier.wait()

class SingleDeviceThread(Thread):
    """
    @brief Worker thread responsible for executing a single data-processing script.
    
    Algorithm: Neighborhood state reduction and update.
    """
    
    def __init__(self, device, script_list, neighbours, index):
        Thread.__init__(self)
        self.device = device
        self.script_list = script_list
        self.neighbours = neighbours
        self.index = index

    def run(self):
        """
        @brief Worker execution loop.
        
        Logic: Attempts to claim a task from the shared list and executes it 
        using neighborhood consensus logic.
        """
        # Synchronization: Critical section for thread-safe task extraction from the device's batch.
        if self.script_list != []:
            # Logic: Pulls next available (script, location) pair.
            (script, location) = self.script_list.pop(self.index)
            self.compute(script, location)

    def update(self, result, location):
        """
        @brief Broadcasts computed results back to the neighborhood.
        """
        for device in self.neighbours:
            device.set_data(location, result)
        self.device.set_data(location, result)

    def collect(self, location, neighbours, script_data):
        """
        @brief Aggregates current data state from all peers for a specific location.
        """
        # Block Logic: Distributed data gathering.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

    def compute(self, script, location):
        """
        @brief Executes the transformation script with neighborhood-wide data.
        """
        # Synchronization: Mutual exclusion for the target data address across the entire network.
        self.device.map_locations[location].acquire()
        
        script_data = []
        self.collect(location, self.neighbours, script_data)

        if script_data != []:
            # Functional Intent: Execute user-defined logic on aggregated data.
            result = script.run(script_data)
            self.update(result, location)

        self.device.map_locations[location].release()

class ReusableBarrierSem():
    """
    @brief Custom reusable synchronization barrier implemented using semaphores.
    
    Algorithm: Two-phase (turnstile) synchronization.
    Logic: Uses double-gating to ensure all threads rendezvous before release 
    and that the barrier is fully reset before the next epoch begins.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with a fixed participant count.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads

        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until the collective rendezvous is reached.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Entrance phase: Accumulates arriving threads and releases the group.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Logic: The final thread releases the entire group.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Exit phase: Ensures all threads have cleared phase 1 before resetting.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
