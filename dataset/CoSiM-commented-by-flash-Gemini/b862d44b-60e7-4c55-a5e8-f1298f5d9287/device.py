

"""
@brief This module defines `Device`, `MyThread` (a script worker thread), and `DeviceThread` classes
for simulating a distributed system.
@details It leverages an external `ReusableBarrier` for inter-device synchronization and employs
a main device thread that dispatches scripts to worker threads for concurrent processing. Access
to location-specific sensor data is managed with a pre-allocated list of locks (`locations_lock`),
optimizing concurrent data processing within the simulation.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrier # Assuming barrier.py contains ReusableBarrier


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It can receive and queue scripts for execution, which are then processed concurrently by its
    dedicated `DeviceThread` by spawning `MyThread` instances. Synchronization across devices is
    managed by a shared `ReusableBarrier`, and thread-safe access to per-location sensor data
    is ensured by a pre-allocated list of `Lock` objects (`locations_lock`).
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, utilizing worker threads for parallel script execution
    and granular locking for data consistency and integrity.
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

        self.barrier = ReusableBarrier(0) # Placeholder barrier, will be updated during setup.

        self.locations_lock = []     # List of Locks for location-specific data access.

        self.thread_list = []        # List to hold MyThread worker instances.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives (barrier and location-specific locks) for all devices.
        @details This method initializes a single `ReusableBarrier` instance and a pre-allocated list of `Lock` objects
        (`locations_lock`) by device 0. These resources are then distributed among all other devices in the simulation.
        Finally, it starts the `DeviceThread` for each device.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of shared synchronization
                     and mutual exclusion primitives.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `self.barrier` refers to a globally shared `ReusableBarrier` instance after setup.
                   `self.locations_lock` is a shared list of `Lock` objects, ensuring consistent access to locations.
                   All `DeviceThread`s are started.
        """
        # Block Logic: Initialize shared barrier and location locks only once by device 0.
        # Invariant: Global `barrier` and `locations_lock` become initialized and shared.
        barrier = ReusableBarrier(len(devices)) # Create a new reusable barrier, sized for all devices.

        if self.device_id == 0: # Only device 0 is responsible for initializing and distributing shared resources.
            # Block Logic: Determine the maximum location ID to pre-allocate `locations_lock`.
            # Invariant: `no_locations` holds the maximum location key + 1.
            locations = []
            for device in devices:
                if device is not None:
                    # Assuming sensor_data keys are integers representing locations.
                    locations.append(max(device.sensor_data.keys()))
            no_locations = max(locations) + 1 # Calculate the number of locks needed.

            # Block Logic: Pre-allocate locks for all possible locations.
            # Invariant: `self.locations_lock` is populated with `no_locations` Lock objects.
            for i in xrange(no_locations): # Using xrange (Python 2 syntax) for loop.
                self.locations_lock.append(Lock())

            # Block Logic: Distribute the newly created barrier and location locks to all devices.
            # Invariant: All devices in `devices` receive a reference to the shared `self.barrier` and `self.locations_lock`.
            for device in devices:
                if device is not None:
                    device.barrier = barrier
                    for i in xrange(no_locations): # Distribute each lock individually.
                        device.locations_lock.append(self.locations_lock[i])
                    device.thread.start() # Start the DeviceThread for each device.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location.
        @details If a script is provided, it's appended to the device's script queue, and the
        `script_received` event is set. If no script is provided (i.e., `script` is None),
        it signifies that the current timepoint's script assignment is complete, and the
        `timepoint_done` event is set to unblock the `DeviceThread`.
        @param script (object): The script object to be executed, or None to signal end of assignments.
        @param location (str): The location associated with the script or data.
        @block_logic Handles the assignment of new scripts or signals the completion of script assignment for a timepoint.
        @pre_condition `self.scripts` is a list, `self.script_received` and `self.timepoint_done` are Event objects.
        @invariant Either a script is added and `script_received` is set, or `timepoint_done` is set.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the queue.
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
        It's assumed that external synchronization (e.g., through `MyThread`'s locks)
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



class MyThread(Thread):
    """
    @brief A worker thread dedicated to executing a single assigned script for a Device instance.
    @details This thread processes a specific script for a given location, collects data from the
    parent device and its neighbors, executes the script's logic, and updates sensor data
    in a thread-safe manner using a location-specific lock from the parent `Device`'s `locations_lock` list.
    @architectural_intent Enhances parallelism by allowing multiple scripts to run concurrently,
    with controlled resource access through location-specific locks to prevent race conditions
    during data manipulation.
    """
    
    def __init__(self, device, neighbours, script, location):
        """
        @brief Initializes a new MyThread instance.
        @param device (Device): The parent Device object that this script thread serves.
        @param neighbours (list): A list of neighboring Device objects from which to collect sensor data.
        @param script (object): The script object to be executed.
        @param location (str): The numerical index of the location associated with the script for which data is processed.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
        self.device = device # Reference to the parent Device object.
        self.neighbours = neighbours # List of neighboring devices.
        self.script = script # The script to execute.
        self.location = location # The numerical index of the sensor data location this script pertains to.

    def run(self):
        """
        @brief The main execution logic for MyThread.
        @details This method acquires a location-specific lock from `self.device.locations_lock`
        to control exclusive access to data at its assigned `location`. It collects data from
        the parent device and its neighbors for the specified location, executes the assigned script,
        and then updates the relevant sensor data for the device and its neighbors. Finally, it
        releases the location-specific lock.
        @block_logic Processes a single script for a specific location, ensuring thread-safe data access.
        @pre_condition `self.script` is an object with a `run` method, `self.device.locations_lock`
                       contains a Lock at index `self.location`.
        @invariant The script is executed, and data is updated under the protection of a location lock.
        """
        # Block Logic: Acquire a lock for the specific location to ensure exclusive access to its data.
        # Invariant: Only one MyThread can modify or read data for `self.location` at a time.
        self.device.locations_lock[self.location].acquire()

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
        if script_data: # Check if script_data is not empty.
            result = self.script.run(script_data) # Execute the script with the collected data.

            # Block Logic: Propagate the script's result to neighboring devices.
            # Invariant: All neighbors receive the updated data for the given location.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Functional Utility: Update the current device's own data with the script's result.
            self.device.set_data(self.location, result)

        self.device.locations_lock[self.location].release() # Release the lock for the current location.



class DeviceThread(Thread):
    """
    @brief The main worker thread for a Device instance.
    @details This thread orchestrates the device's operational cycle for each timepoint.
    It fetches neighbor information, waits for script assignments, and then processes
    these scripts concurrently by spawning temporary `MyThread` worker threads. It ensures
    inter-device synchronization through a shared `ReusableBarrier`.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    abstracting multi-threaded script execution through temporary worker threads and
    coordinating with the global barrier to ensure proper progression of the distributed system.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
        self.device = device # Reference to the parent Device object.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        @details This method continuously monitors the simulation state. For each timepoint,
        it retrieves neighbor information from the supervisor. If neighbors are available,
        it waits until `timepoint_done` is set (signaling that script assignments are complete),
        then processes the assigned scripts by creating and starting `MyThread` instances.
        After all `MyThread` instances complete, it clears `timepoint_done`, resets the script list,
        and finally synchronizes with other `DeviceThread` instances via the global `ReusableBarrier`.
        The loop terminates when the supervisor signals the end of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        concurrent script execution via temporary threads, and inter-device synchronization.
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

            # Block Logic: Create and append temporary worker threads to `self.device.thread_list`.
            # Invariant: Each script from `self.device.scripts` is assigned to a `MyThread` object.
            for (script, location) in self.device.scripts:
                self.device.thread_list.append(MyThread(self.device, neighbours, script, location))

            # Block Logic: Start all temporary worker threads concurrently.
            # Invariant: All temporary threads begin their `run` method concurrently.
            for thread in self.device.thread_list:
                thread.start()

            # Block Logic: Wait for all temporary worker threads to complete their execution.
            # Invariant: The DeviceThread will not proceed until all its temporary worker threads have finished.
            for thread in self.device.thread_list:
                thread.join()

            # Functional Utility: Clear the list of temporary worker threads for the next timepoint.
            self.device.thread_list = [] # Reset thread_list.

            # Functional Utility: Clear the `timepoint_done` event and scripts list for the next timepoint.
            self.device.timepoint_done.clear()
            self.device.scripts = [] # Reset scripts list for the next timepoint.
            
            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances in the simulation will reach this barrier before any proceeds to the next timepoint.
            self.device.barrier.wait()

