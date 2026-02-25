"""
@870af61d-5492-46b2-8fff-6c81700a26bc/device.py
@brief Implements a simulated device for a distributed sensor network,
       including multithreaded script execution and synchronized data processing.

This module defines the architecture for individual devices in a sensor
network. Each device manages its sensor data, receives and executes scripts,
and collaborates with a supervisor and neighboring devices. It leverages
threading primitives like Events, Threads, and Locks for concurrent
script processing and a reusable barrier for global synchronization across devices.
"""


from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.

    Each device has a unique ID, stores sensor data, executes assigned scripts,
    and collaborates with other devices for synchronized data processing.
    It manages its own script queue and communicates its state to a supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's sensor readings,
                            keyed by location.
        @param supervisor: A reference to the supervisor object for inter-device
                           communication and network topology information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        # List to store assigned scripts, each being a (script_object, location) tuple.
        self.scripts = []
        # Event to signal when the current timepoint's script processing is done.
        self.timepoint_done = Event()
        # The main processing thread for this device. Initialized in set_barrier.
        self.thread = None
        # Dictionary to store locks for specific data locations. Managed by a central lock.
        self.locations_locks = None
        # Global lock to protect access to `locations_locks` dictionary.
        self.lock = None
        # Reference to all devices in the network, set during setup.
        self.devices = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization and locking mechanisms across all devices.

        This method should be called by a central orchestrator. It initializes
        a single ReusableBarrier for all devices and a shared structure for
        managing location-specific locks, assigning these to all devices.

        @param devices: A list of all Device objects in the network.
        """
        # Block Logic: This section ensures that a single barrier and a shared
        #              mechanism for managing location locks are created and
        #              distributed among all devices. This logic is only
        #              executed by the "root" device (the device with the maximum ID).
        root = max(devices) # Identify the device with the maximum ID as the "root" for setup.
        if self == root: # Ensure setup logic runs only once.
            map_locks = {} # Shared dictionary for all location-specific locks.
            global_lock = Lock() # Global lock to protect 'map_locks' dictionary itself.
            barrier = ReusableBarrier(len(devices)) # A single barrier for all devices.
            
            # Assign the shared barrier, location locks dictionary, and global lock to all devices.
            for device in devices:
                device.set_barrier(barrier)
                device.set_locations_locks(map_locks)
                device.set_lock(global_lock)
        self.devices = devices # Store the list of all devices.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.

        If a script is provided, it's added to the device's script queue. An event
        is set to notify the device's processing thread that new scripts are available.
        If `script` is None, it signals the end of the current timepoint for script assignment.

        @param script: The script object to be executed, or None.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal the DeviceThread that new scripts arrived.
        else:
            self.timepoint_done.set() # Signal that no more scripts are coming for this timepoint.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The location for which to retrieve data.
        @return: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The location at which to set the data.
        @param data: The new data value to be set.
        """

        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its processing thread to complete.
        """

        self.thread.join()

    def set_barrier(self, barrier):
        """
        @brief Sets the shared ReusableBarrier and starts the device's processing thread.

        @param barrier: The ReusableBarrier instance shared among all devices.
        """
        self.thread = DeviceThread(self, barrier)
        self.thread.start()

    def set_locations_locks(self, locations_locks):
        """
        @brief Sets the shared dictionary for location-specific locks.

        @param locations_locks: A dictionary where keys are locations and values are Lock objects.
        """
        self.locations_locks = locations_locks

    def set_lock(self, lock):
        """
        @brief Sets the global lock used to protect access to the `locations_locks` dictionary.

        @param lock: The threading.Lock instance.
        """
        self.lock = lock

    def acquire_location(self, location):
        """
        @brief Acquires a lock for a specific data location.

        This ensures exclusive access to the data at `location` across all devices.
        If a lock for the location doesn't exist, it's created.

        @param location: The data location for which to acquire the lock.
        """
        location = str(location) # Ensure location is a string for dictionary key.
        self.lock.acquire() # Acquire global lock to protect `locations_locks` dictionary.
        # Block Logic: If a lock for the given location doesn't exist, create it.
        if (location in self.locations_locks) is False:
            self.locations_locks[location] = Lock()
        self.locations_locks[location].acquire() # Acquire the specific location lock.
        self.lock.release() # Release global lock.

    def release_location(self, location):
        """
        @brief Releases the lock for a specific data location.

        @param location: The data location for which to release the lock.
        """
        location = str(location) # Ensure location is a string for dictionary key.
        self.locations_locks[location].release()


class DeviceThread(Thread):
    """
    @brief The main processing thread for a Device.

    This thread manages script execution for its associated device. It coordinates
    with other devices using a barrier, fetches neighbors' data, and dispatches
    scripts for processing in separate ScriptThreads.
    """

    def __init__(self, device, barrier):
        """
        @brief Initializes the DeviceThread with a reference to its parent device and a shared barrier.

        @param device: The Device instance this thread is associated with.
        @param barrier: The ReusableBarrier instance used for synchronization.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously orchestrates script processing: it waits for new scripts,
        dispatches them to `ScriptThread`s, waits for all `ScriptThread`s to complete
        for a timepoint, and then synchronizes with other device threads using a barrier.
        The loop terminates if the supervisor signals no more neighbors (end of simulation).
        """

        # Block Logic: Main loop for continuous operation of the DeviceThread.
        while True:
            # Get the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If supervisor returns None for neighbors, it indicates simulation end.
            if neighbours is None:
                break # Exit the main device thread loop if no more neighbors.

            # Synchronization Point: Wait for all devices to reach this barrier before proceeding.
            self.barrier.wait()
            
            self.device.timepoint_done.wait() # Wait until the current timepoint is marked as done.
            self.device.timepoint_done.clear() # Clear the timepoint_done event for the next cycle.
            
            threads = [] # List to hold ScriptThread objects.

            # Block Logic: Iterate through assigned scripts and launch ScriptThreads for processing.
            for (script, location) in self.device.scripts:
                script_data = [] # List to hold data collected for the script.

                # Acquire the lock for the specific sensor location before accessing data.
                self.device.acquire_location(location)

                # Block Logic: Collect data from neighboring devices at the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Collect data from the local device at the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If any script data was collected, launch a ScriptThread.
                if script_data != []:
                    thread = ScriptThread(script, script_data, self.device, neighbours, location)
                    thread.start() # Start the ScriptThread.
                    threads.append(thread) # Add to the list of active script threads.
                else:
                    # If no script data, release the location lock immediately.
                    self.device.release_location(location)

            # Block Logic: Wait for all launched ScriptThreads to complete their execution.
            for thread in threads:
                thread.join()


class ScriptThread(Thread):
    """
    @brief A worker thread responsible for executing a single script for a Device.

    ScriptThreads gather data, run the script's logic, and update sensor data
    for the local device and its neighbors, ensuring thread-safe access to locations.
    """
    
    def __init__(self, script, data, device, neighbours, location):
        """
        @brief Initializes a ScriptThread.

        @param script: The script object to execute.
        @param data: The collected sensor data to be passed to the script.
        @param device: The parent Device object.
        @param neighbours: A list of neighboring Device objects.
        @param location: The data location pertinent to this script.
        """
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.script = script
        self.data = data
        self.device = device
        self.neighbours = neighbours
        self.location = location

    def run(self):
        """
        @brief The main execution logic for the ScriptThread.

        It runs the assigned script with the provided data, then updates the
        sensor data for the local device and its neighbors, finally releasing
        the location lock.
        """
        result = self.script.run(self.data) # Execute the script with the collected data.

        # Block Logic: Update sensor data for all neighboring devices.
        for device in self.neighbours:
            device.set_data(self.location, result)

        # Block Logic: Update sensor data for the local device.
        self.device.set_data(self.location, result)
        # Release the lock for the sensor location after all updates are done.
        self.device.release_location(self.location)
