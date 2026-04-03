"""
@8cfcdc1d-a15a-4646-b00b-163050a2b8b1/device.py
@brief This script implements device behavior for a distributed system simulation,
featuring a multi-threaded architecture with global and local synchronization barriers,
and granular locking for data access. Devices manage sensor data, execute scripts,
and coordinate with a supervisor and neighboring devices, ensuring thread-safe operations.
Domain: Concurrency, Distributed Systems, Simulation, Thread Synchronization, Data Management.
"""

from threading import Event, Thread, Lock, Semaphore
from collections import deque


class Device(object):
    """
    @brief Represents a single device in the simulated distributed system.
    This version manages its own sensor data, executes scripts using worker threads,
    and coordinates synchronization through global and local barriers, as well as
    location-specific locks managed through a shared 'zones' dictionary.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data for various locations.
        @param supervisor: A reference to the supervisor object for coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Inline: Event to signal that a new script has been assigned to the device.
        self.script_received = Event()
        # Inline: List to store assigned scripts and their locations.
        self.scripts = []
        # Inline: Event to signal that all scripts for the current timepoint have been processed.
        self.timepoint_done = Event()
        
        # Inline: List to hold references to DeviceThread worker instances.
        self.thread = []
        
        # Inline: Local lock for protecting device-specific shared resources within a Device instance.
        self.local_lock = Lock()
        
        # Inline: A shared dictionary (across all devices) to store locks for different data locations/zones.
        self.zones = None
        # Inline: The number of worker threads each device will spawn.
        self.num_threads = 8
        # Inline: Event to signal that neighbor information has been updated.
        self.got_neighbours = Event()


        # Inline: Event to signal that new scripts have been received and prepared.
        self.got_scripts = Event()
        # Inline: List to store references to neighboring devices.
        self.neighbours = []
        
        # Inline: Lock protecting access to the shared 'zones' dictionary.
        self.zones_lock = None
        
        # Inline: A barrier for synchronizing local worker threads within this device.
        self.local_barrier = ReusableBarrier(self.num_threads)
        
        # Inline: A global barrier for synchronizing all worker threads across all devices.
        self.global_barrier = ReusableBarrier(1) # Initialized with 1, will be updated during setup
        # Inline: A deque (double-ended queue) to manage scripts that need to be processed.
        self.todo_scripts = None

        # Block Logic: Create and start 'num_threads' worker threads for this device.
        for _ in range(self.num_threads):
            self.thread.append(DeviceThread(self))


        for i in range(self.num_threads):
            self.thread[i].start()

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        @return: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device with a list of other devices in the system.
        This method is responsible for setting up shared synchronization primitives
        (global barrier, zones dictionary, and zones lock) across all devices.
        @param devices: A list of all Device objects in the simulation.
        """
        
        # Inline: A shared dictionary (across all devices) to store locks for different data locations/zones.
        zones = {}
        
        # Inline: A global barrier for synchronizing all worker threads across all devices.
        global_barrier = ReusableBarrier(devices[0].num_threads * len(devices))
        
        # Inline: A lock to protect access to the shared 'zones' dictionary.
        zones_lock = Lock()
        # Block Logic: Distribute the shared synchronization objects to all devices.
        for dev in devices:
            dev.zones = zones
            dev.global_barrier = global_barrier
            dev.zones_lock = zones_lock

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location for this device.
        Scripts for the same location are queued. If no script is provided, it signals
        the end of scripts for the current timepoint.
        @param script: The script object to be executed.
        @param location: The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The location identifier for which to retrieve data.
        @return: The sensor data for the location, or None if the location is not present.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
        @param location: The location identifier for which to set data.
        @param data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining all its worker threads.
        """
        for thread in self.thread:
            thread.join()


class DeviceThread(Thread):
    """
    @brief A worker thread for a Device. Each device can have multiple such threads.
    These threads are responsible for fetching neighbors, waiting for scripts,
    processing them, and coordinating synchronization at both local and global levels.
    """

    def __init__(self, device):
        """
        @brief Initializes a DeviceThread instance.
        @param device: The parent Device object that this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for a DeviceThread. It continuously synchronizes
        with other threads/devices, fetches neighbors, processes assigned scripts,
        and manages data locks.
        """
        # Inline: Signal that this thread is ready to receive neighbor information.
        self.device.got_neighbours.set()
        # Inline: Signal that this thread is ready to receive scripts.
        self.device.got_scripts.set()
        
        # Block Logic: Continuous operational loop for the worker thread.
        while True:
            # Block Logic: Global synchronization point. All worker threads across all devices wait here.
            self.device.global_barrier.wait()

            # Block Logic: Local critical section for updating neighbor information.
            self.device.local_lock.acquire()
            # Pre-condition: If 'got_neighbours' event is set, meaning it's time to update neighbors.
            if self.device.got_neighbours.isSet():
                # Inline: Clear the timepoint_done event for the new timepoint.
                self.device.timepoint_done.clear()
                
                # Inline: Fetch updated neighbors from the supervisor.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                # Inline: Clear the 'got_neighbours' event to prevent redundant updates.
                self.device.got_neighbours.clear()

            # Post-condition: Release the local lock.
            self.device.local_lock.release()

            # Block Logic: Local barrier for worker threads within the same device.
            self.device.local_barrier.wait()
            # Inline: Re-set 'got_neighbours' event for the next cycle.
            self.device.got_neighbours.set()
            # Pre-condition: If neighbors is None, it indicates simulation termination.
            if self.device.neighbours is None:
                break

            # Block Logic: Wait for the Device's timepoint_done event, signaling scripts are ready.
            self.device.timepoint_done.wait()

            # Block Logic: Local critical section for preparing scripts.
            self.device.local_lock.acquire()
            # Pre-condition: If 'got_scripts' event is set, meaning it's time to load new scripts.
            if self.device.got_scripts.isSet():
                # Inline: Populate the 'todo_scripts' deque with scripts assigned to the device.
                self.device.todo_scripts = deque(self.device.scripts)
                # Inline: Clear the 'got_scripts' event to prevent redundant loading.
                self.device.got_scripts.clear()
            self.device.local_lock.release()
            
            # Block Logic: Local barrier for worker threads within the same device, ensuring scripts are ready.
            self.device.local_barrier.wait()
            # Inline: Re-set 'got_scripts' event for the next cycle.
            self.device.got_scripts.set()

            # Block Logic: Process scripts from the 'todo_scripts' deque.
            while True:
                self.device.local_lock.acquire()
                
                # Pre-condition: Check if there are any scripts left to process in the deque.
                if not self.device.todo_scripts:
                    
                    self.device.local_lock.release()
                    break

                # Inline: Pop the next script and its location from the deque.
                (script, location) = self.device.todo_scripts.popleft()

                # Block Logic: Acquire lock for the 'zones' dictionary to manage location-specific locks.
                self.device.zones_lock.acquire()
                # Pre-condition: If no lock exists for this specific location, create one.
                if location not in self.device.zones.keys():
                    self.device.zones[location] = Lock()
                self.device.zones_lock.release()

                # Block Logic: Acquire the location-specific lock to ensure exclusive access to data at this location.
                self.device.zones[location].acquire()
                # Post-condition: Release the local lock, as zone lock is now held.
                self.device.local_lock.release()

                script_data = []
                # Block Logic: Collect data from neighboring devices for the specified location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collect data from the current device for the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: Execute the script only if there is data to process.
                if script_data != []:
                    # Inline: Execute the script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Update data on neighboring devices with the script's result.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Update data on the current device with the script's result.
                    self.device.set_data(location, result)
                # Post-condition: Release the location-specific lock.
                self.device.zones[location].release()


class ReusableBarrier():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.
    It allows a fixed number of threads to wait for each other before proceeding
    together, and can be reused multiple times. This is a basic semaphore-based implementation.
    Algorithm: Double-phase semaphore-based barrier.
    Time Complexity: O(N) for each `wait` call where N is the number of threads, due to semaphore releases.
    Space Complexity: O(1) for internal state.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # Inline: Counter for the first phase of the barrier. Using a list to allow modification within methods.
        self.count_threads1 = [self.num_threads]
        # Inline: Counter for the second phase of the barrier.
        self.count_threads2 = [self.num_threads]
        # Inline: Lock to protect access to the thread counters.
        self.count_lock = Lock()                 
        # Inline: Semaphore for synchronizing threads in the first phase.
        self.threads_sem1 = Semaphore(0)         
        # Inline: Semaphore for synchronizing threads in the second phase.
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all participating threads
        have reached this point.
        """
        # Block Logic: Execute the first phase of the barrier synchronization.
        self.phase(self.count_threads1, self.threads_sem1)
        # Block Logic: Execute the second phase of the barrier synchronization.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the barrier synchronization.
        @param count_threads: The counter list for the current phase.
        @param threads_sem: The semaphore for the current phase.
        """
        # Block Logic: Acquire lock to safely decrement the thread counter.
        with self.count_lock:
            count_threads[0] -= 1
            # Pre-condition: If this is the last thread to reach the barrier in this phase.
            if count_threads[0] == 0:            
                # Block Logic: Release the semaphore 'num_threads' times to unblock all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()        
                # Inline: Reset the counter for the next use of this phase of the barrier.
                count_threads[0] = self.num_threads  
        # Post-condition: Acquire the semaphore, waiting if not yet released by the last thread.
        threads_sem.acquire()                    
                                                 
