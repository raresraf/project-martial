
"""
@brief This module defines the Device, ScriptThread, and DeviceThread classes, representing simulated devices
in a distributed system with an advanced threading model.
@details This system includes script execution in dedicated script threads and location-specific locks
for managing shared sensor data, enabling more complex and concurrent simulations.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    @brief Represents a simulated device within a distributed sensor network.
    @details This class models an individual device, managing sensor data, interacting with a supervisor,
    and coordinating the execution of assigned scripts. It employs a shared `ReusableBarrierSem` for global
    synchronization across all devices and uses `ScriptThread` instances for parallel script execution.
    It also implements location-specific locks to manage concurrent access to sensor data.
    @architectural_intent Acts as an autonomous agent, processing data, and coordinating with other devices
    and a supervisor, with enhanced concurrency for script processing and data management.
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
        self.script_received = Event() # Event to signal when new scripts have been assigned.
        self.scripts = []            # List to store assigned scripts and their locations.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's processing by the supervisor.
        self.thread = DeviceThread(self) # The main worker thread for this device.
        self.thread.start()          # Start the device's execution thread.
        self.locationlocks = {}      # Dictionary to hold locks for each sensor data location.
        self.lock = Lock()           # General lock for protecting shared resources within the device (e.g., sensor_data updates).
        self.bariera = None          # Reference to the shared ReusableBarrierSem for inter-device synchronization.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources and synchronization mechanisms for all devices.
        @details This method initializes a global `ReusableBarrierSem` and a shared dictionary of `locationlocks`.
        These resources are distributed among all devices, ensuring proper synchronization and data integrity.
        It specifically ensures that the barrier and location locks are created only once by device 0
        and then shared with all other devices.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of shared synchronization and locking primitives.
        @pre_condition `devices` is a list of `Device` instances, and this method is called once for each device.
        @invariant `self.bariera` is a shared `ReusableBarrierSem` instance for all devices, and `self.locationlocks`
                   is a shared dictionary of `Lock` objects, ensuring concurrent access to sensor data is managed.
        """
        bariera = ReusableBarrierSem(len(devices)) # Create a new reusable barrier for all devices.
        locations = []                             # List to collect all unique sensor locations across all devices.
        
        # Block Logic: Distribute the shared barrier and collect all unique sensor locations.
        # Invariant: Each device receives a reference to the global barrier, and `locations` contains all unique sensor keys.
        for dev in devices:
            if (self.device_id == 0): # Only device 0 is responsible for creating and distributing the barrier.
                dev.bariera = bariera
            for location in dev.sensor_data:
                if not location in locations:
                    locations.append(location)
        
        # Block Logic: Initialize and distribute location-specific locks.
        # Invariant: `self.locationlocks` becomes a shared dictionary of locks, with a lock for each unique location.
        if (self.device_id == 0): # Only device 0 initializes the shared location locks.
            for location in locations:
                self.locationlocks[location] = Lock() # Create a lock for each unique location.
            for dev in devices:
                dev.locationlocks = self.locationlocks # Distribute the shared location locks to all devices.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location.
        @details If a script is provided, it's appended to the device's script queue, and the `script_received` event is set.
        If no script is provided (i.e., `script` is None), it signifies that the current timepoint's script assignment is complete,
        and the `timepoint_done` event is set, unblocking the `DeviceThread`.
        @param script (object): The script object to be executed, or None to signal end of assignments for a timepoint.
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
        @details This method acquires a general device lock before modifying the sensor data
        to ensure thread safety during updates.
        @param location (str): The location whose data is to be updated.
        @param data (object): The new data value for the specified location.
        @block_logic Safely updates the internal sensor data using a lock to prevent race conditions.
        @pre_condition `self.sensor_data` is a dictionary, and `self.lock` is an initialized Lock object.
        @invariant If `location` is a key in `self.sensor_data`, its value is updated under lock protection.
        """
        self.lock.acquire() # Acquire the general device lock to protect sensor_data modifications.
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release() # Release the lock after the update.

    def shutdown(self):
        """
        @brief Shuts down the device by joining its associated thread.
        @details This ensures that the device's worker thread completes its execution before the program exits.
        """
        self.thread.join()


class ScriptThread(Thread):
    """
    @brief A worker thread responsible for executing a subset of assigned scripts for a Device instance.
    @details This thread processes scripts for specific locations, collects data from neighbors and
    itself, executes the script's logic, and updates sensor data in a thread-safe manner using
    location-specific locks.
    @architectural_intent Improves parallelism by allowing multiple scripts to run concurrently
    within a single Device's timepoint processing, optimizing data processing throughput.
    """
    
    def __init__(self, device, scripts, locations, neighbours):
        """
        @brief Initializes a new ScriptThread instance.
        @param device (Device): The parent Device object that this script thread serves.
        @param scripts (list): A list of script objects to be executed by this thread.
        @param locations (list): A list of locations, corresponding to each script, for which data is processed.
        @param neighbours (list): A list of neighboring Device objects from which to collect sensor data.
        """
        Thread.__init__(self) # Initialize the base Thread class.
        self.device = device
        self.scripts = scripts
        self.locations = locations
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for the script thread.
        @details This method iterates through its assigned scripts and locations. For each script,
        it acquires a location-specific lock, collects data from the parent device and its neighbors,
        executes the script, and then updates the relevant sensor data for the device and its neighbors
        before releasing the lock.
        @block_logic Processes a batch of scripts for specific locations, ensuring thread-safe data access.
        @pre_condition `self.device` and `self.neighbours` are valid, `self.scripts` and `self.locations` are aligned lists.
        @invariant Each script is executed, and data is updated under the protection of a `locationlock`.
        """
        i = 0 # Initialize an index to track current script and location.
        # Block Logic: Iterate through each script assigned to this thread.
        # Invariant: Each script is processed, and its results are propagated.
        for script in self.scripts:
            # Block Logic: Acquire a lock for the specific location to ensure exclusive access to its data.
            # Pre-condition: `self.device.locationlocks` contains a Lock object for `self.locations[i]`.
            # Invariant: Only one script thread can modify or read data for `self.locations[i]` at a time.
            self.device.locationlocks[self.locations[i]].acquire()
            script_data = [] # List to accumulate data for the current script's execution.
            
            # Block Logic: Collect data from neighboring devices for the current location.
            # Invariant: `script_data` will contain data from all available neighbors for the given location.
            for device in self.neighbours:
                data = device.get_data(self.locations[i])
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Collect data from the current device itself for the current location.
            # Invariant: If available, the device's own data for the location is added to `script_data`.
            data = self.device.get_data(self.locations[i])
            if data is not None:
                script_data.append(data)
            
            # Block Logic: Execute the script if there is any data to process.
            # Pre-condition: `script` is an object with a `run` method, and `script_data` is a list of data.
            # Invariant: `result` holds the output of the script's execution.
            if script_data != []:
                result = script.run(script_data) # Execute the script with the collected data.
                
                # Block Logic: Propagate the script's result to neighboring devices.
                # Invariant: All neighbors receive the updated data for the given location.
                for device in self.neighbours:
                    device.set_data(self.locations[i], result)
                
                # Functional Utility: Update the current device's own data with the script's result.
                self.device.set_data(self.locations[i], result)
            
            self.device.locationlocks[self.locations[i]].release() # Release the lock for the current location.
            i += 1 # Move to the next script and location.


class DeviceThread(Thread):
    """
    @brief The main worker thread for a Device instance.
    @details This thread orchestrates the overall timepoint processing for its associated Device.
    It manages synchronization using a shared barrier, waits for script assignments from the supervisor,
    and then distributes these scripts among multiple `ScriptThread` instances for parallel execution.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    abstracting the multi-threaded script execution to optimize performance for complex workloads.
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
        it retrieves neighbor information, waits for script assignment completion from the supervisor,
        and then processes the assigned scripts. Script processing is parallelized by delegating
        subsets of scripts to multiple `ScriptThread` instances. After all `ScriptThread` instances
        complete, it synchronizes with other `DeviceThread` instances via the shared barrier.
        The loop terminates when the supervisor signals the end of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        script parallelization, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `timepoint_done` event, and a shared `bariera` (ReusableBarrierSem).
        @invariant The thread progresses through timepoints, processes scripts concurrently,
                   and ensures global synchronization.
        """
        tlist = [] # List to hold ScriptThread instances for parallel script execution.
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
            
            # Functional Utility: Reset the event for the next timepoint's script assignment.
            self.device.timepoint_done.clear()
            
            # Block Logic: Initialize a fixed pool of ScriptThread instances for parallel processing.
            # Invariant: `tlist` contains 8 new, unstarted ScriptThread objects.
            for index in range(8): # Creates 8 ScriptThread instances. This number might be configurable.
                tlist.append(ScriptThread(self.device, [], [], neighbours))
            index = 0 # Index to distribute scripts among the ScriptThread instances.
            
            # Block Logic: Distribute the assigned scripts among the ScriptThread instances in `tlist`.
            # Pre-condition: `self.device.scripts` contains tuples of (script, location).
            # Invariant: Each script from `self.device.scripts` is appended to one of the `ScriptThread` instances' lists.
            for (script, location) in self.device.scripts:
                tlist[index].scripts.append(script)
                tlist[index].locations.append(location)
                index = (index + 1) % 8 # Round-robin distribution of scripts.
            
            # Block Logic: Start all ScriptThread instances to execute scripts in parallel.
            # Invariant: All ScriptThread instances begin their `run` method concurrently.
            for thread in tlist:
                    thread.start()
            
            # Block Logic: Wait for all ScriptThread instances to complete their execution.
            # Invariant: The DeviceThread will not proceed until all its ScriptThread children have finished.
            for thread in tlist:
                    thread.join()
            
            # Functional Utility: Clear the list of ScriptThreads for the next timepoint.
            del tlist[:]
            
            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances will reach this barrier before any proceeds to the next timepoint.
            self.device.bariera.wait()
