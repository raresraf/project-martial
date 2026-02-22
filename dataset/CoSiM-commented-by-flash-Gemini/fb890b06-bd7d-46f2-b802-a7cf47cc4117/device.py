

"""
@brief This module defines `Device`, `ScriptThread`, and `DeviceThread` classes for simulating a distributed system.
@details It utilizes a `ReusableBarrierSem` for inter-device synchronization, employs a main device thread that
dispatches scripts to worker script threads for concurrent processing, and dynamically manages location-specific
locks for thread-safe sensor data access. This design aims to balance concurrency with controlled resource
management within a distributed sensor network simulation.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It can receive and queue scripts for execution, which are then processed by its dedicated
    `DeviceThread` that spawns `ScriptThread` instances for concurrent processing.
    Synchronization across devices is managed by a shared `ReusableBarrierSem`, and thread-safe
    access to per-location sensor data is ensured by dynamically managed `locks` (a dictionary
    of `Lock` objects for each sensor location).
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, leveraging worker threads for parallel script execution
    and dynamic granular locking for data consistency and integrity.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary containing initial sensor data,
                                   where keys are locations and values are data readings.
        @param supervisor (object): A reference to the supervisor object that manages
                                    the overall distributed system and provides access
                                    to network information (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are ready for execution.
        self.scripts = []            # List to store assigned scripts and their locations.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's script assignment.
        self.thread = DeviceThread(self) # The main worker thread for this device.
        self.thread.start()          # Start the device's execution thread.
        self.barrier = None          # Reference to the shared ReusableBarrierSem for inter-device synchronization.
        self.devices = []            # Reference to the list of all Device objects in the simulation.
        self.locks = {}              # Dictionary to hold locks for each sensor data location (shared across devices).
        self.lock_used = None        # Flag to indicate if a lock for a location has been found/used.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared `ReusableBarrierSem` for all devices.
        @details This method initializes a single `ReusableBarrierSem` instance and distributes it
        to all devices in the simulation. It stores a reference to all devices and assigns the
        shared barrier to each of them.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of the shared barrier.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `self.barrier` refers to a globally shared `ReusableBarrierSem` instance after setup.
                   `self.devices` holds references to all simulation devices.
        """
        self.devices = devices # Store the list of all devices.
        # Functional Utility: Create a new reusable barrier, sized for all devices.
        barrier = ReusableBarrierSem(len(devices))

        # Block Logic: Assign the newly created barrier to all devices.
        # Invariant: Each device in `devices` receives a reference to the shared barrier.
        for device in devices:
            device.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location, and ensures a lock exists for it.
        @details If a script is provided, it's appended to the device's script queue. This method also checks
        if a lock for the specified `location` already exists in `self.locks` (potentially from another device),
        and if not, creates a new `Lock` for it. The `script_received` event is set to signal the `DeviceThread`.
        If no script is provided (`script` is None), it signifies that the current timepoint's script assignment
        is complete, and the `timepoint_done` event is set to unblock the `DeviceThread`.
        @param script (object): The script object to be executed, or None to signal end of assignments.
        @param location (str): The location associated with the script or data.
        @block_logic Handles script assignment and ensures existence of a location-specific lock for thread safety.
        @pre_condition `self.scripts` is a list, `self.script_received` and `self.timepoint_done` are Event objects.
                       `self.locks` is a dictionary for location-specific locks.
        @invariant Either a script is added and `script_received` is set, or `timepoint_done` is set.
                   A lock for `location` is ensured to exist in `self.locks`.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the queue.
            
            # Block Logic: Check if a lock for the location already exists across devices, or create a new one.
            # This logic suggests `locks` is intended to be a globally shared dictionary initialized on first use.
            self.lock_used = None
            for device in self.devices: # Iterate through all devices to see if a lock for this location already exists.
                if device.locks.get(location) is not None:
                    self.locks[location] = device.locks[location] # Use the existing lock.
                    self.lock_used = 1 # Mark that a lock was found.
                    break # Exit loop as lock is found/assigned.

            if self.lock_used is None: # If no existing lock was found for this location.
                self.locks[location] = Lock() # Create a new lock for this location.

            self.lock_used = None # Reset flag.
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # Signal that script assignments for the current timepoint are complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location (str): The location for which to retrieve data.
        @return object: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @details This method updates the internal sensor data if the location exists.
        It's assumed that external synchronization (e.g., through `ScriptThread`'s locks)
        protects this operation during concurrent modifications.
        @param location (str): The location whose data is to be updated.
        @param data (object): The new data value for the specified location.
        @block_logic Updates the internal sensor data.
        @pre_condition `self.sensor_data` is a dictionary.
        @invariant If `location` is a key in `self.sensor_data`, its value is updated.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its associated thread.
        @details This ensures that the device's worker thread completes its execution before the program exits.
        """
        self.thread.join()


class ScriptThread(Thread):
    """
    @brief A worker thread dedicated to executing a single assigned script for a Device instance.
    @details This thread processes a specific script for a given location, collects data from the
    parent device and its neighbors, executes the script's logic, and updates sensor data
    in a thread-safe manner using a location-specific lock acquired from the parent `Device`.
    @architectural_intent Enhances parallelism by allowing multiple scripts to run concurrently,
    with controlled resource access through location-specific locks to prevent race conditions
    during data manipulation.
    """
    
    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes a new ScriptThread instance.
        @param device (Device): The parent Device object that this script thread serves.
        @param script (object): The script object to be executed.
        @param location (str): The location associated with the script for which data is processed.
        @param neighbours (list): A list of neighboring Device objects from which to collect sensor data.
        """
        Thread.__init__(self) # Initialize the base Thread class.
        self.device = device # Reference to the parent Device object.
        self.script = script # The script to execute.
        self.location = location # The sensor data location this script pertains to.
        self.neighbours = neighbours # List of neighboring devices.

    def run(self):
        """
        @brief The main execution logic for the ScriptThread.
        @details This method acquires a location-specific lock from `self.device.locks` to control
        exclusive access to data at its assigned location. It collects data from the parent device
        and its neighbors, executes the assigned script, and then updates the relevant sensor data
        for the device and its neighbors. Finally, it releases the location-specific lock.
        @block_logic Processes a single script for a specific location, ensuring thread-safe data access.
        @pre_condition `self.script` is an object with a `run` method, `self.device.locks`
                       contains a Lock for `self.location`.
        @invariant The script is executed, and data is updated under the protection of a location lock.
        """
        # Block Logic: Acquire a lock for the specific location to ensure exclusive access to its data.
        # Invariant: Only one ScriptThread can modify or read data for `self.location` at a time.
        with self.device.locks[self.location]: # Uses `with` statement for automatic lock acquisition and release.
            script_data = [] # List to accumulate data for the current script's execution.
            
            # Block Logic: Collect data from neighboring devices for the current location.
            # Invariant: `script_data` will contain data from all available neighbors for the given location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Collect data from the current device itself for the current location.
            # Invariant: If available, the device's own data for the location is added to `script_data`.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Execute the script if there is any data to process.
            # Pre-condition: `self.script` is an object with a `run` method, and `script_data` is a list of data.
            # Invariant: `result` holds the output of the script's execution.
            if script_data != []:
                result = self.script.run(script_data) # Execute the script with the collected data.

                # Block Logic: Propagate the script's result to neighboring devices.
                # Invariant: All neighbors receive the updated data for the given location.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                # Functional Utility: Update the current device's own data with the script's result.
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    @brief The main worker thread for a Device instance.
    @details This thread orchestrates the device's operational cycle, including
    fetching neighbor information, waiting for script assignments, and then dispatching
    these scripts to individual `ScriptThread` instances for concurrent execution.
    It manages synchronization through a shared barrier and ensures that script processing
    is completed before proceeding to the next timepoint.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    abstracting multi-threaded script execution through worker threads and coordinating
    with the global barrier to ensure proper progression of the distributed system.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
        self.device = device # Reference to the parent Device object.
        self.script_threads = [] # List to hold ScriptThread instances.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        @details This method continuously monitors the simulation state. For each timepoint,
        it retrieves neighbor information from the supervisor. If neighbors are available,
        it waits until `timepoint_done` is set (signaling that script assignments are complete),
        then processes the assigned scripts by creating and starting `ScriptThread` instances.
        After all script threads complete, it clears `timepoint_done`, resets the script list,
        and finally synchronizes with other `DeviceThread` instances via the shared `ReusableBarrierSem`.
        The loop terminates when the supervisor signals the end of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        parallel script execution, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `scripts` list, `timepoint_done` event, and `barrier`.
        @invariant The thread progresses through timepoints, processes scripts concurrently,
                   and ensures global synchronization.
        """
        while True:
            # Functional Utility: Get information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # Block Logic: Check if the simulation should terminate.
            # Pre-condition: `neighbours` list indicates the current state of the network.
            # Invariant: The loop terminates if no neighbors are returned by the supervisor.
            if neighbours is None:
                break

            # Block Logic: Wait until script assignments for the current timepoint are complete.
            # Pre-condition: `self.device.timepoint_done` is an Event object.
            # Invariant: The thread proceeds only after the supervisor signals completion of script assignment.
            self.device.timepoint_done.wait()

            # Block Logic: Create and start ScriptThread instances for each assigned script.
            # Invariant: Each script leads to the creation and start of a `ScriptThread`.
            for (script, location) in self.device.scripts:
                thread = ScriptThread(self.device, script, location, neighbours)
                self.script_threads.append(thread) # Add the thread to the list for joining later.

            # Block Logic: Start all spawned ScriptThread instances concurrently.
            # Invariant: All `ScriptThread` instances begin their `run` method concurrently.
            for thread in self.script_threads:
                thread.start()
            # Block Logic: Wait for all spawned ScriptThread instances to complete their execution.
            # Invariant: The DeviceThread will not proceed until all its ScriptThread children have finished.
            for thread in self.script_threads:
                thread.join()
            
            # Functional Utility: Clear the list of ScriptThreads for the next timepoint.
            self.script_threads = [] # Reset script_threads list for the next timepoint.

            # Functional Utility: Clear the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()
            # Functional Utility: Clear the scripts list for the next timepoint.
            self.device.scripts = [] # Reset scripts list for the next timepoint.
            
            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances will reach this barrier before any proceeds to the next timepoint.
            self.device.barrier.wait()

