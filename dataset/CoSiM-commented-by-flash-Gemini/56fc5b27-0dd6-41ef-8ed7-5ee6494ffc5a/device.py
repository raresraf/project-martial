"""
@56fc5b27-0dd6-41ef-8ed7-5ee6494ffc5a/device.py
@brief Implements a simulated device for a distributed sensor network, with concurrent script execution, dynamic lock management, and barrier synchronization.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` that dispatches scripts to `ScriptThread` instances,
limiting concurrency with a `Semaphore`. Synchronization is handled by a `ReusableBarrierSem`,
and data access is protected by dynamically shared, location-specific `Lock` objects.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, a shared barrier, and dynamically managed
    location-specific locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing the device's local sensor readings.
        @param supervisor: The supervisor object responsible for managing the overall simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal that a script has been assigned.
        self.scripts = [] # List to store assigned scripts (tuples of (script_object, location)).
        self.thread = DeviceThread(self)
        self.barrier = None # Shared barrier for global time step synchronization.
        self.threads = [] # List to keep track of active `ScriptThread` instances.
        self.semaphore = Semaphore(8) # Semaphore to limit the number of concurrent `ScriptThread`s to 8.
        self.lock = {} # Dictionary to hold `Lock` objects for each sensor data location.
        self.all_devices = [] # Will store a reference to all devices in the simulation.
        self.thread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared `ReusableBarrierSem` for synchronization among all devices.
        If not already set, it initializes the global barrier and distributes it to all devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: Stores a reference to all devices if not already set.
        if len(self.all_devices) == 0:
            self.all_devices = devices

        # Block Logic: Initializes the shared barrier if it has not been set yet, and distributes it to all devices.
        # Invariant: A single `ReusableBarrierSem` instance is created and shared across all devices.
        if self.barrier is None:
            barrier = ReusableBarrierSem(len(devices))
            for device in devices:
                device.barrier = barrier

    def update_locks(self, scripts):
        """
        @brief Dynamically creates and shares location-specific locks across all devices.
        For each script and its associated location, if a lock for that location doesn't exist,
        a new `Lock` object is created and then distributed to all devices' `self.lock` dictionary.
        @param scripts: A list of scripts and their locations (`[(script_obj, location_key)]`).
        Precondition: `self.all_devices` must be populated.
        """
        # Block Logic: Iterates through assigned scripts to ensure each location has a shared lock.
        for (_, location) in scripts:
            # Block Logic: If a lock for the current location does not exist, create one and share it.
            if not self.lock.has_key(location): # Using has_key for compatibility (can use `location not in self.lock`).
                self.lock[location] = Lock()
                for device in self.all_devices:
                    device.lock[location] = self.lock[location]


    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Signals that a script has been received after assignment.
        @param script: The script object to assign, or `None` to signal completion.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: Signals that script assignments are complete.
            self.script_received.set()


    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its graceful completion.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts concurrently via `ScriptThread` instances (with a limited thread pool),
    and coordinating with other device threads using a shared barrier.
    Time Complexity: O(T * S_total * (N * D_access + D_script_run)) where T is the number of timepoints,
    S_total is the total number of scripts executed by the device, N is the number of neighbors,
    D_access is data access time, and D_script_run is script execution time.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Waits for the `script_received` event to be set, indicating that scripts are ready to be processed.
        3. Dynamically updates the shared `Lock` objects for all relevant data locations.
        4. Synchronizes with all other device threads using a shared barrier (`ReusableBarrierSem`).
        5. Creates and starts `ScriptThread` instances for each assigned script, limiting concurrency
           using a `Semaphore` (thread pool).
           Invariant: At most 8 `ScriptThread` instances run concurrently.
        6. Waits for all `ScriptThread` instances to complete.
        7. Clears the `script_received` event for the next timepoint.
        8. Synchronizes again with all other device threads using the shared barrier.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the device has received all scripts for the current timepoint.
            self.device.script_received.wait()

            # Block Logic: Dynamically updates/creates shared locks for all locations involved in the current scripts.
            self.device.update_locks(self.device.scripts)

            # Block Logic: Synchronizes all device threads at the start of script execution phase.
            self.device.barrier.wait()

            # Block Logic: Creates and starts `ScriptThread` instances for each assigned script,
            # using a semaphore to limit concurrent executions (thread pool).
            for script in self.device.scripts:
                thread = ScriptThread(self.device, neighbours, script)
                self.device.threads.append(thread)
                
                # Block Logic: Acquires a semaphore permit before starting a new script thread.
                self.device.semaphore.acquire()
                thread.start()


            # Block Logic: Waits for all initiated `ScriptThread` instances to complete their execution.
            for thread in self.device.threads:
                thread.join()

            self.device.threads = [] # Clears the list of threads.

            # Block Logic: Clears the `script_received` event for the next timepoint cycle.
            self.device.script_received.clear()

            # Block Logic: Synchronizes all device threads again after script execution and cleanup,
            # ensuring all devices are ready to proceed to the next timepoint.
            self.device.barrier.wait()

class ScriptThread(Thread):
    """
    @brief A dedicated thread for executing a single script for a specific data location.
    This thread is responsible for gathering data, running the script, and then
    propagating the results to relevant devices, ensuring thread-safe access to data
    through location-specific locks.
    """
    
    def __init__(self, device, neighbours, script):
        """
        @brief Initializes a `ScriptThread` instance.
        @param device: The parent `Device` instance for which the script is being run.
        @param neighbours: A list of neighboring `Device` instances.
        @param script: The script object to execute (tuple of `(script_object, location_key)`).
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script # `script` is expected to be a tuple `(script_object, location_key)`.

    def run(self):
        """
        @brief The main execution logic for `ScriptThread`.
        Block Logic:
        1. Acquires the location-specific lock for the current script's data location.
        2. Collects data from neighboring devices and its own device for the specified `location`.
        3. Executes the assigned `script` if any data was collected.
        4. Propagates the script's `result` to neighboring devices and its own device.
        5. Releases the location-specific lock.
        6. Releases a permit back to the global semaphore, allowing another script thread to start.
        Invariant: All data access and modification for a given `location` are protected by a shared `Lock`.
        """

        # Block Logic: Acquires the location-specific lock before accessing or modifying data for this location.
        self.device.lock.get(self.script[1]).acquire()

        script_data = []
        
        # Block Logic: Collects data from neighboring devices for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.script[1]) # self.script[1] is the location key.
            if data is not None:
                script_data.append(data)

        # Block Logic: Collects data from its own device for the specified location.
        data = self.device.get_data(self.script[1])
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script if any data was collected and propagates the result.
        if script_data != []:
            
            result = self.script[0].run(script_data) # self.script[0] is the script object.

            # Block Logic: Updates neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.script[1], result)
                
            # Block Logic: Updates its own device's data with the script's result.
            self.device.set_data(self.script[1], result)

        # Block Logic: Releases the location-specific lock after all data operations for this script are complete.
        self.device.lock.get(self.script[1]).release()

        # Block Logic: Releases a permit back to the global semaphore, allowing another script thread to start.
        self.device.semaphore.release()
