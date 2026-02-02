"""
@file device.py
@brief This module defines the `Device` and `DeviceThread` classes, which simulate distributed computation
across multiple devices, each processing scripts over shared sensor data. It manages synchronization
and data consistency for concurrent operations.

Algorithm:
- Each `Device` hosts multiple `DeviceThread`s.
- `DeviceThread`s run concurrently, processing assigned scripts at specific timepoints.
- Synchronization is managed using `ReusableBarrier` for thread coordination within a device and across all devices.
- Data access to sensor data (local and remote/neighbour) is controlled via threading.Lock mechanisms to ensure consistency.
- Scripts are partitioned among threads within a device for parallel execution.

Time Complexity:
- `Device.setup_devices`: O(D * L) where D is the number of devices and L is the number of locations per device, to initialize locks.
- `DeviceThread.run`: The core loop runs indefinitely, processing timepoints. Within each timepoint, script processing depends on the number of scripts (S) and neighbours (N), leading to O(S/T * N) where T is threads per device.
Space Complexity:
- `Device`: O(L) for sensor data, O(T) for threads, O(S) for scripts, O(D * L) for global locks.
- `DeviceThread`: O(1) beyond references to the parent device and shared resources.
"""


from __future__ import division


from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier
from math import ceil


class Device(object):
    """
    @class Device
    @brief Represents a single computational device in a distributed system.
    Functional Utility: Manages sensor data, assigned scripts, and coordinates multiple
    `DeviceThread`s to perform parallel processing over timepoints, interacting with a supervisor
    and neighbouring devices for data exchange.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id (int): A unique identifier for this device.
        @param sensor_data (dict): A dictionary representing sensor data local to this device.
                                  Keys are location IDs, values are data.
        @param supervisor (object): A reference to the supervisor object responsible for
                                    global coordination, such as retrieving neighbour information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        
        self.timepoint_done = Event() # Event to signal when all threads within this device have completed a timepoint.

        
        
        self.neighbours = None # List of neighbouring devices, initialized to None and set by the supervisor.

        
        self.scripts = [] # List of (script, location) tuples assigned to this device for execution.

        
        self.threads = [] # List of DeviceThread instances belonging to this device.

        
        
        
        self.l_loc_dev = {} # Dictionary to store locks for specific location data across devices.

        
        
        
        self.l_all_threads = None # Global lock for controlling access to shared data structures across all threads.

        
        
        self.b_all_threads = None # Barrier for synchronizing all threads across all devices.

        
        
        
        
        
        
        
        b_local = ReusableBarrier(8) # Local barrier for synchronizing the 8 threads within this device.

        
        
        
        e_local = Event() # Local event to signal within the device's threads.

        
        
        for i in xrange(8): # Block Logic: Initialize and start 8 DeviceThread instances.
            thread = DeviceThread(self, i, b_local, e_local) # Create a new DeviceThread.
            self.threads.append(thread) # Add the thread to the device's list.
            thread.start() # Start the thread's execution.


    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return (str): A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id


    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives across all devices in the system.
        Functional Utility: Initializes a global barrier and a shared dictionary of locks
        for inter-device and inter-thread data access, ensuring that these resources are
        only set once by the first device in the list.
        @param devices (list): A list of all Device instances in the simulation.
        """
        if devices[0] == self: # Block Logic: Ensure setup is performed only once by the first device.
            nr_of_threads = sum([len(device.threads) for device in devices]) # Calculate total number of threads across all devices.
            barrier = ReusableBarrier(nr_of_threads) # Create a global barrier for all threads.
            loc_dev_lock = {
                (device.device_id, location_id): Lock() # Create a lock for each (device, location) pair.
                for device in devices
                for location_id in device.sensor_data
                }

            set_data_lock = Lock() # Create a global lock for setting data.

            for device in devices: # Block Logic: Assign the shared resources to all devices.
                device.b_all_threads = barrier # Assign the global barrier.
                device.l_loc_dev = loc_dev_lock # Assign the shared location data locks.
                device.l_all_threads = set_data_lock # Assign the global data lock.


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location, or signals completion.
        Functional Utility: Adds a script-location pair to the device's queue, or marks
        the current timepoint as done if no more scripts are to be assigned.
        @param script (object): The script object to assign. Can be None to signal completion.
        @param location (object): The location ID where the script should be executed.
        """
        if script is not None: # Block Logic: If a script is provided, add it to the list.
            self.scripts.append((script, location))

        else: # Block Logic: If script is None, signal that the timepoint is done.
            self.timepoint_done.set()


    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location (object): The location ID of the sensor data to retrieve.
        @return (object or None): The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data: # Block Logic: Check if the location exists in local sensor data.
            return self.sensor_data[location]

        return None # Return None if location not found.


    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
        @param location (object): The location ID where the sensor data should be set.
        @param data (object): The new sensor data value.
        """
        if location in self.sensor_data: # Block Logic: Check if the location exists in local sensor data.
            self.sensor_data[location] = data


    def access_data(self, location):
        """
        @brief Acquires locks necessary to safely access data at a specific location, including neighbour data.
        Functional Utility: Ensures exclusive access to sensor data by acquiring locks before modification.
        This includes acquiring the global lock (`l_all_threads`) and specific locks for the target
        location on both the current device and its neighbours.
        @param location (object): The location ID for which data access is requested.
        """
        self.l_all_threads.acquire() # Acquire the global lock for all threads.

        if location in self.sensor_data: # Block Logic: Acquire local lock if location exists.
            self.l_loc_dev[(self.device_id, location)].acquire()

        for device in self.neighbours: # Block Logic: Iterate through neighbours to acquire remote locks.
            if device != self and location in device.sensor_data: # Only acquire if not self and location exists.
                device.l_loc_dev[(device.device_id, location)].acquire()
        self.l_all_threads.release() # Release the global lock.


    def release_data(self, location):
        """
        @brief Releases locks after data access at a specific location.
        Functional Utility: Releases the locks acquired by `access_data` on the current device
        and its neighbours, allowing other threads to access the data.
        @param location (object): The location ID for which data locks are to be released.
        """
        if location in self.sensor_data: # Block Logic: Release local lock if location exists.
            self.l_loc_dev[(self.device_id, location)].release()

        for device in self.neighbours: # Block Logic: Iterate through neighbours to release remote locks.
            if device != self and location in device.sensor_data: # Only release if not self and location exists.
                device.l_loc_dev[(device.device_id, location)].release()


    def shutdown(self):
        """
        @brief Shuts down all threads associated with this device.
        Functional Utility: Waits for all DeviceThread instances to complete their execution,
        ensuring a clean termination of the device's operations.
        """
        for thread in self.threads: # Block Logic: Join each thread to wait for its completion.
            thread.join()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief Represents a single thread of execution within a `Device`.
    Functional Utility: Executes assigned scripts concurrently, handling synchronization
    with other threads within the same device and across all devices. It fetches
    neighbour information, processes scripts by accessing and modifying shared sensor data.
    """

    def __init__(self, device, id_thread, barrier, event):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The parent Device instance to which this thread belongs.
        @param id_thread (int): A unique identifier for this thread within its device.
        @param barrier (ReusableBarrier): A local barrier for synchronizing threads within the device.
        @param event (Event): A local event for signaling within the device's threads.
        """
        Thread.__init__(
            self,
            name="Device Thread {0}-{1}".format(device.device_id, id_thread) # Set a descriptive thread name.
            )

        self.device = device
        self.id_thread = id_thread
        self.barrier = barrier
        self.event = event


    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Continuously processes timepoints, synchronizes with other threads,
        fetches neighbour data (for the first thread in a device), partitions and executes assigned
        scripts, and handles data access/release with appropriate locking mechanisms.
        """
        while True: # Block Logic: Main loop for continuous timepoint processing.

            
            
            if self.device.threads[0] == self: # Block Logic: Only the first thread of the device performs global updates.
                
                
                self.device.neighbours = self.device.supervisor.get_neighbours() # Fetch neighbours from the supervisor.

                
                self.event.set() # Signal that neighbours have been updated.

            else:
                
                
                self.event.wait() # Wait for the first thread to update neighbours.


            
            if self.device.neighbours is None: # Block Logic: If neighbours are still None, it's a shutdown signal.
                break # Exit the loop and terminate the thread.

            
            self.device.timepoint_done.wait() # Wait for the signal that scripts for the current timepoint are assigned.

            
            self.barrier.wait() # Wait at the local barrier for all threads within this device to reach this point.

            
            
            
            
            if self.device.threads[0] == self: # Block Logic: Only the first thread of the device resets for the next timepoint.
                self.device.timepoint_done.clear() # Clear the timepoint_done event.
                self.event.clear() # Clear the local event.

            
            partition_size = int(ceil( # Block Logic: Calculate partition size for script distribution among threads.
                len(self.device.scripts) /
                len(self.device.threads)
                ))

            
            down_lim = self.id_thread * partition_size # Calculate lower bound for script partition.
            up_lim = min(down_lim + partition_size, len(self.device.scripts)) # Calculate upper bound for script partition.

            # Block Logic: Iterate through the assigned script partition.
            # Invariant: Each thread processes its dedicated subset of scripts for the current timepoint.
            for (script, location) in self.device.scripts[down_lim : up_lim]:

                
                # Precondition: Locks for location and its neighbours should be acquired before accessing data.
                # Functional Utility: Acquires necessary locks to ensure thread-safe access to sensor data.
                self.device.access_data(location)

                script_data = [] # List to accumulate data for the script.

                
                
                for device in self.device.neighbours: # Block Logic: Collect data from neighbouring devices.
                    data = device.get_data(location) # Get data from neighbour.
                    if data is not None:
                        script_data.append(data) # Add to script data if available.

                
                data = self.device.get_data(location) # Block Logic: Collect data from the current device.
                if data is not None:
                    script_data.append(data) # Add to script data if available.

                if script_data != []: # Block Logic: If there is data, run the script and update results.

                    
                    # Functional Utility: Executes the assigned script with the collected data.
                    result = script.run(script_data)

                    
                    # Functional Utility: Propagates the script's result to neighbouring devices and the current device.
                    for device in self.device.neighbours: # Update neighbour data.
                        device.set_data(location, result)

                    self.device.set_data(location, result) # Update local data.

                
                
                # Postcondition: Locks acquired for location and its neighbours should be released after data access.
                # Functional Utility: Releases the locks, making the data available for other threads.
                self.device.release_data(location)


            
            
            # Synchronization Point: All threads across all devices wait here after processing their scripts.
            # Functional Utility: Ensures that no thread proceeds to the next timepoint before all others have finished.
            self.device.b_all_threads.wait()