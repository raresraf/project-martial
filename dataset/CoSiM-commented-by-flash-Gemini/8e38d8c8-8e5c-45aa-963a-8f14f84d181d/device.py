"""
@8e38d8c8-8e5c-45aa-963a-8f14f84d181d/device.py
@brief This script implements device behavior for a distributed system simulation,
featuring a multi-threaded architecture with a custom reusable barrier and a detailed
approach to managing location-specific data locks across devices. It orchestrates
the parallel execution of individual scripts.
Domain: Concurrency, Distributed Systems, Simulation, Thread Synchronization, Parallel Processing.
"""

from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """
    @brief Represents a single device in the simulated distributed system.
    This version manages its own sensor data, executes scripts using worker threads
    for individual script processing, and coordinates synchronization through a
    global barrier and shared location-specific locks.
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
        # Inline: The main thread for the device's operational logic.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Inline: Reference to the shared ReusableBarrierSem.
        self.barrier = None
        # Inline: A shared dictionary that maps location IDs to Lock objects for thread-safe access.
        self.map_locations = None # Initialized by device 0 in setup_devices.


    def __str__(self):
        """
        @brief Returns a string representation of the device.
        @return: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device with a list of other devices in the system.
        This method is primarily responsible for initializing and distributing
        shared synchronization primitives (barrier and location locks) across all devices.
        Only the device with the lowest device_id (usually 0) performs the initialization.
        @param devices: A list of all Device objects in the simulation.
        """
        
        flag = True # Inline: Flag to determine if this device has the lowest device_id.
        device_number = len(devices)

        # Block Logic: Determine if this device has the lowest device_id.
        for dev in devices:
            if self.device_id > dev.device_id:
                flag = False

        # Block Logic: If this device has the lowest device_id, it initializes the shared resources.
        if flag == True:
            # Inline: Create a shared reusable barrier for all devices.
            barrier = ReusableBarrierSem(device_number)
            # Inline: A dictionary to store location-specific locks, shared across all devices.
            map_locations = {}
            tmp = {} # Inline: Temporary dictionary, not explicitly used as intended.
            # Block Logic: Initialize a lock for each unique sensor data location found across all devices.
            for dev in devices:
                dev.barrier = barrier # Assign the shared barrier to each device.
                # Inline: Identify new locations from the current device's sensor data not yet in map_locations.
                tmp = list(set(dev.sensor_data) - set(map_locations))
                # Block Logic: For each new unique location, create a new Lock.
                for i in tmp:
                    map_locations[i] = Lock()
                dev.map_locations = map_locations # Assign the shared map of locks to each device.
                tmp = {} # Inline: Reset tmp.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location for this device.
        If a script is provided, it's added to the device's script list. If no script
        is provided (script is None), it signals that the current timepoint is done.
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
        @brief Shuts down the device by joining its main thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main thread responsible for a Device's operational loop.
    It manages the lifecycle of script execution for its parent device,
    fetches neighbors, and orchestrates the parallel execution of scripts
    via `SingleDeviceThread`s.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device: The Device object that this thread manages.
        """
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        @brief The main loop for the device thread. It continuously fetches neighbors,
        waits for timepoint signals, and dispatches script execution to `SingleDeviceThread`s.
        """
        
        # Block Logic: Continuous operational loop for the device thread.
        while True:
            # Inline: Clear the timepoint_done event for the new timepoint.
            self.device.timepoint_done.clear()
            # Block Logic: Fetch the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If supervisor returns None, it indicates simulation termination.
            if neighbours is None:
                break
            # Block Logic: Wait until the current timepoint's scripts are ready to be processed.
            self.device.timepoint_done.wait()
            
            # Inline: Create a list to temporarily store scripts for processing.
            script_list = []
            # Inline: List to hold references to `SingleDeviceThread` instances.
            thread_list = []
            index = 0 # Inline: This index seems to be used incorrectly in a pop-based script processing.
            
            # Block Logic: Populate `script_list` with all scripts assigned to the device.
            for script in self.device.scripts:
                script_list.append(script)
            
            # Block Logic: Spawn 8 `SingleDeviceThread`s to process scripts.
            # Flaw: The `pop(self.index)` in `SingleDeviceThread.run` combined with `index = 0` here
            # and a fixed number of threads will likely lead to incorrect processing or IndexError.
            for i in xrange(8): # xrange is used for efficiency in older Python versions
                thread = SingleDeviceThread(self.device, script_list, neighbours, index)
                thread.start()
                thread_list.append(thread)
            
            # Block Logic: Wait for all `SingleDeviceThread`s to complete their execution.
            for i in xrange(len(thread_list)):
                thread_list[i].join()
            
            # Block Logic: Synchronize with other devices via the shared barrier.
            self.device.barrier.wait()

class SingleDeviceThread(Thread):
    """
    @brief A worker thread designed to execute a single script for a device,
    collecting data from neighbors and updating relevant data locations.
    It uses location-specific locks for thread-safe data access.
    """
    
    def __init__(self, device, script_list, neighbours, index):
        """
        @brief Initializes a SingleDeviceThread instance.
        @param device: The parent Device object.
        @param script_list: A list of scripts from which this thread will attempt to pop one.
        @param neighbours: A list of neighboring Device objects.
        @param index: An index, likely intended to specify which script to process (but used with pop which removes).
        """
        Thread.__init__(self)
        self.device = device
        self.script_list = script_list
        self.neighbours = neighbours
        self.index = index

    def run(self):
        """
        @brief Executes its assigned script. It collects data, runs the script,
        updates data, and ensures thread-safe access using location-specific locks.
        """
      
        # Block Logic: If there are scripts available, attempt to pop and process one.
        # Flaw: Using `pop(self.index)` with `self.index=0` across multiple threads
        # will cause race conditions and incorrect script distribution.
        if self.script_list != []:
            (script, location) = self.script_list.pop(self.index) # Flaw: Race condition and incorrect index usage.
            self.compute(script, location)

    def update(self, result, location):
        """
        @brief Updates the sensor data on all neighboring devices and the current device
        with the computed result for a specific location.
        @param result: The result computed by the script.
        @param location: The location for which data is being updated.
        """
        # Block Logic: Update data on all neighboring devices.
        for device in self.neighbours:
            device.set_data(location, result)
        # Block Logic: Update data on the current device.
        self.device.set_data(location, result)

    def collect(self, location, neighbours, script_data):
        """
        @brief Collects sensor data for a given location from neighboring devices and itself,
        ensuring thread-safe access to the location's data.
        @param location: The location from which to collect data.
        @param neighbours: A list of neighboring Device objects.
        @param script_data: A list to append the collected data to.
        """
        # Block Logic: Acquire the location-specific lock to ensure exclusive access during data collection.
        self.device.map_locations[location].acquire()
        # Block Logic: Collect data from neighboring devices.
        for device in self.neighbours:
            
            data = device.get_data(location)
            if data is None:
                pass # Inline: Skip if no data for this location.
            else:
                script_data.append(data) # Inline: Add collected data to the script_data list.

        # Block Logic: Collect data from the current device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

    def compute(self, script, location):
        """
        @brief Orchestrates the data collection, script execution, and data update for a single script.
        @param script: The script object to be executed.
        @param location: The data location the script operates on.
        """
        script_data = []
        self.collect(location, self.neighbours, script_data)

        # Pre-condition: Only execute the script if data was successfully collected.
        if script_data == []:
            pass # Inline: If no data, do nothing.
        else:
            # Inline: Execute the script with the collected data.
            result = script.run(script_data)
            self.update(result, location)

        # Post-condition: Release the location-specific lock.
        self.device.map_locations[location].release()

class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads using semaphores.
    It ensures that all participating threads pause at a specific point until every thread
    reaches that point, then they all proceed.
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
        # Inline: Counter for the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # Inline: Counter for the second phase of the barrier.
        self.count_threads2 = self.num_threads


        # Inline: Lock to protect access to the thread counters.
        self.counter_lock = Lock()
        # Inline: Semaphore for synchronizing threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        # Inline: Semaphore for synchronizing threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all participating threads
        have reached this point. It coordinates two phases of synchronization.
        """
        # Block Logic: Execute the first phase of the barrier synchronization.
        self.phase1()
        # Block Logic: Execute the second phase of the barrier synchronization.
        self.phase2()

    def phase1(self):
        """
        @brief Implements the first phase of the barrier synchronization.
        Threads decrement a counter and the last one releases all others via a semaphore.
        """
        # Block Logic: Acquire lock to safely decrement the thread counter for phase 1.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Pre-condition: If this is the last thread to reach the barrier in phase 1.
            if self.count_threads1 == 0:
                # Block Logic: Release the semaphore 'num_threads' times to unblock all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Inline: Reset the counter for the next use of this phase of the barrier.
                self.count_threads1 = self.num_threads
        # Post-condition: Acquire the semaphore, waiting if not yet released by the last thread.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Implements the second phase of the barrier synchronization.
        Threads decrement a counter and the last one releases all others via a semaphore.
        """
        # Block Logic: Acquire lock to safely decrement the thread counter for phase 2.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Pre-condition: If this is the last thread to reach the barrier in phase 2.
            if self.count_threads2 == 0:
                # Block Logic: Release the semaphore 'num_threads' times to unblock all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Inline: Reset the counter for the next use of this phase of the barrier.
                self.count_threads2 = self.num_threads
        # Post-condition: Acquire the semaphore, waiting if not yet released by the last thread.
        self.threads_sem2.acquire()
