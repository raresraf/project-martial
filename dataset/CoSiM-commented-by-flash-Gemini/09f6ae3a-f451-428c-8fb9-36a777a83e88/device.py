"""
@file device.py
@brief This module defines the `Device` and `DeviceThread` classes for simulating
distributed computation across multiple devices. It manages sensor data, script execution,
and synchronization among concurrent threads and devices.

Algorithm:
- Each `Device` manages a pool of `DeviceThread`s.
- Threads process a queue of assigned scripts, potentially for different data locations.
- Data access is synchronized using per-location reentrant locks (`RLock`) to ensure consistency
  when accessing local or neighbour's sensor data.
- Timepoint processing is coordinated using `Event` objects and `Barrier`s for intra-device
  and inter-device synchronization.
- Scripts are assigned to a queue, and threads acquire from this queue using a `Semaphore`.

Time Complexity:
- `Device.setup_devices`: O(D * L) to initialize location locks across all devices, where D is the number of devices and L is the number of locations.
- `DeviceThread.run`: The main loop runs indefinitely. Within each timepoint, script processing
  depends on the number of scripts (S) and neighbours (N), leading to operations proportional to
  S * (L_acquire + L_release + N * L_access), where L_acquire/release/access are lock operation times.
Space Complexity:
- `Device`: O(L) for sensor data, O(C) for threads, O(S) for scripts, O(L_total) for location locks (where L_total is sum of all locations across all devices).
- `DeviceThread`: O(1) beyond references to the parent device and shared resources.
"""


from threading import Event, Thread, Semaphore, RLock
from barrier import Barrier


class Device(object):
    """
    @class Device
    @brief Represents a single computational device in a distributed system simulation.
    Functional Utility: Manages its local sensor data, assigns scripts for processing,
    and coordinates its worker threads (`DeviceThread`s). It uses various synchronization
    primitives to ensure data consistency and proper execution flow across timepoints
    and in interaction with neighbouring devices.
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
        self.device_id = device_id # Unique identifier for this device.
        self.sensor_data = sensor_data # Dictionary storing local sensor data.
        self.supervisor = supervisor # Reference to the supervisor object.

        self.scripts = [] # Stores all scripts assigned to this device for a given timepoint.
        self.scripts_queue = [] # Queue for scripts currently awaiting execution by threads.
        self.threads = [] # List of DeviceThread instances managed by this device.
        self.cores_no = 8 # Number of worker threads (cores) for this device.
        self.neighbours = [] # List of neighbouring devices, updated by the supervisor.

        self.timepoint_done = Event() # Event to signal when all scripts for a timepoint have been assigned.
        self.queue_lock = RLock() # Reentrant lock for protecting access to the scripts queue.
        self.location_locks = {} # Dictionary to hold RLocks for individual data locations, ensuring exclusive access.
        self.queue_sem = Semaphore(value=0) # Semaphore to control access to the scripts queue by worker threads.
        self.timepoint_barrier = Barrier() # Global barrier for synchronizing all devices at the end of a timepoint.
        self.neighbours_barrier = Barrier(self.cores_no) # Barrier for synchronizing worker threads within this device.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return (str): A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives across all devices in the system.
        Functional Utility: Initializes global resources such as location-specific locks
        and the timepoint barrier. This setup is crucial for inter-device data consistency
        and global synchronization. It ensures that locks for all possible locations
        (across all devices) are centrally managed.
        @param devices (list): A list of all Device instances in the simulation.
        """
        
        # Block Logic: Initialize location locks for all possible locations across all devices.
        # Precondition: `self.location_locks` is initially empty.
        # Postcondition: `self.location_locks` contains an `RLock` for every unique location ID across all devices.
        for dev in devices:
            for location in dev.sensor_data.keys():
                if location not in self.location_locks:
                    self.location_locks[location] = RLock()

        # Block Logic: Set up the global timepoint barrier and share resources from the first device.
        # Functional Utility: The first device in the list acts as a coordinator for setting up
        # global synchronization primitives which are then shared among all other devices.
        self.timepoint_barrier.set_num_threads(len(devices)*self.cores_no) # Configure global barrier with total threads.
        self.timepoint_barrier = devices[0].timepoint_barrier # Share the global barrier instance.
        self.location_locks = devices[0].location_locks # Share the global dictionary of location locks.

        
        for i in xrange(self.cores_no): # Block Logic: Initialize and start worker threads for this device.
            self.threads.append(DeviceThread(self, i)) # Create a new DeviceThread.
            self.threads[i].start() # Start the thread's execution.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location, or signals the end of a timepoint.
        Functional Utility: Adds a script-location pair to the execution queue, making it available
        for worker threads. If `script` is `None`, it signals that all scripts for the current
        timepoint have been assigned and are ready for processing.
        @param script (object): The script object to assign. If `None`, signals end of timepoint assignment.
        @param location (object): The location ID where the script should be executed.
        """
        
        
        # Block Logic: Handle script assignment or timepoint completion signal.
        if script is not None: # Case 1: A new script is assigned.
            with self.queue_lock: # Acquire reentrant lock to safely modify the queue.
                self.timepoint_done.clear() # Clear the 'timepoint_done' event as new scripts are coming.
                
                
                self.scripts.append((script, location)) # Add script to the full list of scripts for the timepoint.
                self.scripts_queue.append((script, location)) # Add script to the queue for immediate processing.
            self.queue_sem.release() # Release semaphore, indicating a script is available in the queue.
        else: # Case 2: Signal that all scripts for the current timepoint have been assigned.
            with self.queue_lock: # Acquire reentrant lock.
                self.timepoint_done.set() # Set the 'timepoint_done' event.
            
            for _ in xrange(self.cores_no): # Functional Utility: Release the semaphore multiple times
                                           # to unblock all worker threads waiting for scripts.
                self.queue_sem.release()

    def recreate_queue(self):
        """
        @brief Recreates the script execution queue from the full list of scripts for the current timepoint.
        Functional Utility: Prepares the script queue for processing a new timepoint by repopulating
        it with all assigned scripts, ensuring all threads can access them.
        """
        
        
        # Block Logic: Populate `scripts_queue` with all scripts and release semaphores.
        with self.queue_lock: # Acquire reentrant lock to safely modify the queue.
            for script in self.scripts: # Iterate through the full list of scripts.
                self.scripts_queue.append(script) # Add each script back to the execution queue.
                self.queue_sem.release() # Release semaphore for each script, making it available.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location (object): The location ID of the sensor data to retrieve.
        @return (object or None): The sensor data at the specified location, or `None` if not found.
        """
        return self.sensor_data[location] \
               if location in self.sensor_data else None # Return data if location exists, else None.

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
        @param location (object): The location ID where the sensor data should be set.
        @param data (object): The new sensor data value.
        """
        if location in self.sensor_data: # Block Logic: Check if the location exists in local sensor data.
            self.sensor_data[location] = data # Update data if location is local.

    def shutdown(self):
        """
        @brief Shuts down all threads associated with this device.
        Functional Utility: Waits for all `DeviceThread` instances to complete their execution,
        ensuring a clean termination of the device's operations. This typically implies that
        the threads have received a signal to break their main loop.
        """
        for i in xrange(self.cores_no): # Block Logic: Join each worker thread.
            self.threads[i].join() # Wait for thread `i` to finish.


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief Represents a single worker thread of execution within a `Device`.
    Functional Utility: Executes assigned scripts concurrently, synchronizing its actions
    with other threads in the same device and across devices. It handles fetching
    neighbour information, acquiring scripts from a shared queue, and performing
    data access/modification with appropriate locking.
    """

    def __init__(self, device, thread_id):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The parent `Device` instance to which this thread belongs.
        @param thread_id (int): A unique identifier for this thread within its device.
        """
        Thread.__init__(self, name="Device %d Thread %d" % # Set a descriptive thread name.
                        (device.device_id, thread_id))
        self.device = device # Reference to the parent device.
        self.thread_id = thread_id # Unique ID of this thread.


    def run_script(self, script, location):
        """
        @brief Executes a given script at a specific location, handling data access and updates.
        Functional Utility: This method encapsulates the core logic of script execution,
        which involves collecting relevant sensor data from the current and neighbouring devices,
        running the script, and then propagating the results back to the devices.
        @param script (object): The script object to execute, which has a `run` method.
        @param location (object): The location ID for which the script is being run.
        """
        script_data = [] # List to accumulate data for the script.
        
        # Block Logic: Acquire reentrant lock for the specific location to ensure exclusive access during script execution.
        with self.device.location_locks[location]:
            
            # Block Logic: Collect data from neighbouring devices for the given location.
            for dev in self.device.neighbours:
                data = dev.get_data(location) # Retrieve data from neighbour.
                if data is not None:
                    script_data.append(data) # Add data if available.

            
            # Block Logic: Collect data from the current device for the given location.
            data = self.device.get_data(location) # Retrieve local data.
            if data is not None:
                script_data.append(data) # Add local data if available.

            if script_data != []: # Block Logic: If any data was collected, run the script and update data.
                
                # Functional Utility: Execute the script's `run` method with the collected data.
                result = script.run(script_data)

            
                # Block Logic: Propagate the script's result to neighbouring devices.
                for dev in self.device.neighbours:
                    dev.set_data(location, result)
                    
                self.device.set_data(location, result) # Update local sensor data with the result.


    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Manages the thread's lifecycle, including synchronization at timepoints,
        fetching neighbour information (for the leader thread), and continuously acquiring and
        executing scripts from the device's shared queue.
        """
        while True: # Block Logic: Continuous loop to process timepoints until shutdown.
            
            # Synchronization Point: The first thread (thread_id == 0) acts as the leader for device-wide setup.
            if self.thread_id == 0:
                
                # Functional Utility: Recreate the script queue for the new timepoint.
                self.device.recreate_queue()
                # Functional Utility: Fetch updated neighbour information from the supervisor.
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # Synchronization Point: All threads within this device wait here until the leader thread has updated neighbours.
            self.device.neighbours_barrier.wait()

            if self.device.neighbours is None: # Block Logic: Check for shutdown signal (neighbours being None).
                break # Exit the loop if shutdown is signaled.

            
            while True: # Block Logic: Inner loop to acquire and run scripts until the timepoint is done.
                
                # Synchronization Point: Acquire a script from the queue or wait if none are available.
                self.device.queue_sem.acquire() # Decrement semaphore, waits if value is 0.
                self.device.queue_lock.acquire() # Acquire queue lock for safe queue access.
                # Block Logic: Check if the timepoint is done and no scripts are left in the queue.
                if self.device.timepoint_done.is_set() and \
                    len(self.device.scripts_queue) == 0:
                    self.device.queue_lock.release() # Release queue lock.
                    break # Exit script processing for this timepoint.
                else:
                    # Functional Utility: Pop a script-location pair from the queue for execution.
                    (script, location) = self.device.scripts_queue.pop(0)
                self.device.queue_lock.release() # Release queue lock.

                self.run_script(script, location) # Execute the retrieved script.
            # Synchronization Point: All threads within this device wait here until all scripts for the timepoint are processed.
            self.device.timepoint_barrier.wait() # Global barrier for all threads across all devices.