

"""
@brief This module defines `Device` and `DeviceThread` classes for simulating a distributed system.
@details It implements a multi-threaded execution model where each `Device` manages a pool of
`DeviceThread`s that collaboratively process scripts. Synchronization is handled by reusable barriers
(both inter-device and intra-device) and per-location locks for thread-safe data access,
optimizing concurrent data processing within the simulation.
"""

import cond_barrier # Assuming cond_barrier.py contains ReusableBarrier
from threading import Event, Thread, Lock


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It coordinates a fixed pool of `DeviceThread`s to execute assigned scripts in parallel.
    Synchronization is achieved through a shared `ReusableBarrier` (inter-device) and an internal
    `ReusableBarrier` (intra-device) for its worker threads. Access to per-location sensor data
    is protected by `map_locks`.
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, utilizing an internal thread pool for parallel script
    execution and granular locking for data consistency and integrity.
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
        self.threads = []            # List to hold the DeviceThread worker instances for this device.

        self.neighbourhood = None    # Stores the list of neighboring devices for the current timepoint.
        self.map_locks = {}          # Dictionary to hold locks for each sensor data location (shared across devices).
        self.threads_barrier = None  # Internal barrier for synchronizing the DeviceThread workers within this device.
        self.barrier = None          # Global barrier for synchronizing all Devices across the simulation.
        self.counter = 0             # Counter for distributing scripts among internal DeviceThread workers.
        self.threads_lock = Lock()   # Lock to protect the `counter` variable.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives and initializes internal worker threads.
        @details This method initializes the global `ReusableBarrier` (inter-device) only once by device 0,
        distributes it and the `map_locks` to all devices. It also creates a pool of 8 `DeviceThread` workers
        for this device, initializes an internal barrier (`threads_barrier`) for these workers, and starts them.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of shared resources, and internal thread management.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant Global `barrier` and `map_locks` are set, and internal `DeviceThread`s are running.
        """
        # Block Logic: Initialize global barrier and map_locks only once by device 0.
        # Invariant: Global `barrier` and `map_locks` become initialized and shared.
        if self.device_id == 0: # Only device 0 is responsible for initializing and distributing.
            num_threads = len(devices) # Total number of devices in the simulation.
            
            # Functional Utility: Create the global barrier, sized for all devices times 8 internal threads.
            # This implies the barrier synchronizes all `DeviceThread` instances globally.
            self.barrier = cond_barrier.ReusableBarrier(num_threads * 8)

            # Block Logic: Distribute the newly created global barrier and map_locks to all devices.
            # Invariant: All devices have references to the shared `self.barrier` and `self.map_locks`.
            for device in devices:
                device.barrier = self.barrier
                device.map_locks = self.map_locks

        # Block Logic: Initialize internal worker threads and their synchronization barrier.
        # Invariant: `self.threads` contains 8 `DeviceThread` instances, and `self.threads_barrier` is set.
        self.threads_barrier = cond_barrier.ReusableBarrier(8) # Internal barrier for the 8 worker threads of THIS device.
        for i in range(8): # Create 8 DeviceThread workers for this device.
            self.threads.append(DeviceThread(self, i, self.threads_barrier))

        # Block Logic: Start all internal DeviceThread workers.
        # Invariant: All `DeviceThread` instances associated with this `Device` are running.
        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location.
        @details If a script is provided, it's appended to the device's script queue, and the
        `script_received` event is set. If no script is provided (i.e., `script` is None),
        it signifies that the current timepoint's script assignment is complete, and the
        `timepoint_done` event is set to unblock the `DeviceThread` workers.
        It also ensures that a lock exists for the specified location in `map_locks`.
        @param script (object): The script object to be executed, or None to signal end of assignments.
        @param location (str): The location associated with the script or data.
        @block_logic Handles the assignment of new scripts or signals the completion of script assignment for a timepoint.
        @pre_condition `self.scripts` is a list, `self.script_received` and `self.timepoint_done` are Event objects.
        @invariant Either a script is added and `script_received` is set, or `timepoint_done` is set.
                   A lock for `location` is ensured to exist in `self.map_locks`.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the queue.
            self.script_received.set() # Signal that new scripts have been received.
        else:
            self.timepoint_done.set() # Signal that script assignments for the current timepoint are complete.

        # Block Logic: Ensure a lock exists for the given location, initializing it if not present.
        # Invariant: `self.map_locks[location]` holds a valid Lock object.
        if location not in self.map_locks:
            self.map_locks[location] = Lock() # Create a new lock for this location if it doesn't exist.

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
        It's assumed that `map_locks` are used externally to protect concurrent access
        to this data during script execution.
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
        @brief Shuts down the device by joining its associated worker threads.
        @details This ensures that all `DeviceThread` workers complete their execution before the program exits.
        """
        for thread in self.threads:
            thread.join()



class DeviceThread(Thread):
    """
    @brief A worker thread managed by a Device instance, designed to collaboratively process scripts.
    @details Multiple `DeviceThread` instances within a single `Device` work together. This thread
    synchronizes with its peers within the same device using an internal `threads_barrier` and
    with other `Device` instances via a global `barrier`. It fetches scripts, acquires location-specific
    locks, processes data from neighbors and itself, executes the script, and propagates results.
    @architectural_intent Enables fine-grained parallel processing of scripts within a single `Device`
    by dividing the workload among multiple threads, improving throughput while maintaining synchronization
    and data consistency through managed locks and barriers.
    """
    
    def __init__(self, device, id, barrier):
        """
        @brief Initializes a new DeviceThread worker.
        @param device (Device): The parent Device object that this thread serves.
        @param id (int): A unique identifier for this specific worker thread within its parent Device.
        @param barrier (cond_barrier.ReusableBarrier): The internal barrier used to synchronize
                                                       worker threads within the same parent Device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
        self.device = device # Reference to the parent Device object.
        self.id = id         # Unique ID of this worker thread.
        self.thread_barrier = barrier # Internal barrier for this device's worker threads.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread worker.
        @details This method orchestrates the worker's tasks for each timepoint. It handles
        initial fetching of neighbor data (only by `id == 0`), synchronizes with other
        worker threads using `self.thread_barrier`, and then collaboratively processes
        scripts assigned to the parent `Device`. Scripts are picked using a shared counter
        (`self.device.counter`) protected by a lock. For each script, it acquires a
        location-specific lock, collects data, executes the script, and updates data,
        then releases the lock. Finally, it synchronizes with all other `Device` instances
        via the global `self.device.barrier`.
        @block_logic Manages the collaborative and synchronized execution of scripts within a Device.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `scripts` list, `timepoint_done` event, `neighbourhood`, `map_locks`,
                       `threads_barrier`, `barrier`, `counter`, and `threads_lock`.
        @invariant The thread progresses through timepoints, processes scripts collaboratively,
                   and ensures both intra-device and inter-device synchronization.
        """
        while True:
            # Block Logic: Only the first worker thread (id == 0) fetches neighbor information once per timepoint.
            # Invariant: `self.device.neighbourhood` is updated for the current timepoint.
            if self.id == 0:
                self.device.neighbourhood = self.device.supervisor.get_neighbours()

            # Block Logic: Synchronize all worker threads within this device.
            # Invariant: All 8 worker threads of this device reach this point before any proceeds.
            self.thread_barrier.wait()

            # Block Logic: Check if the simulation should terminate based on supervisor's neighbor information.
            # Invariant: The loop terminates if no neighbors are returned by the supervisor.
            if self.device.neighbourhood is None:
                break 

            # Block Logic: Wait until script assignments for the current timepoint are complete for the parent device.
            # Pre-condition: `self.device.timepoint_done` is an Event object.
            # Invariant: The worker threads proceed only after the supervisor signals completion of script assignment.
            self.device.timepoint_done.wait()

            # Block Logic: Collaboratively process assigned scripts. Workers take scripts from a shared queue.
            # Invariant: Each script from `self.device.scripts` is processed exactly once by one of the worker threads.
            while True:
                script = None
                location = None
                # Block Logic: Acquire `threads_lock` to safely get the next script to process from the shared list.
                # Invariant: `self.device.counter` is atomically incremented, and `script`/`location` are assigned.
                with self.device.threads_lock:
                    if self.device.counter == len(self.device.scripts): # Check if all scripts have been distributed.
                        break # All scripts for this timepoint processed.
                    (script, location) = self.device.scripts[self.device.counter] # Get the next script.
                    self.device.counter = self.device.counter + 1 # Increment counter for the next worker.
                
                # Block Logic: Acquire a lock for the specific location to ensure exclusive access to its data.
                # Pre-condition: `self.device.map_locks` contains a Lock object for `location`.
                # Invariant: Only one worker thread can modify or read data for `location` at a time.
                self.device.map_locks[location].acquire()
                script_data = [] # List to accumulate data for the current script's execution.

                # Block Logic: Collect data from neighboring devices for the current location.
                # Invariant: `script_data` will contain data from all available neighbors for the given location.
                for device in self.device.neighbourhood:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Block Logic: Collect data from the current device itself for the current location.
                # Invariant: If available, the device's own data for the location is added to `script_data`.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Execute the script if there is any data to process.
                # Pre-condition: `script` is an object with a `run` method, and `script_data` is a list of data.
                # Invariant: `result` holds the output of the script's execution.
                if script_data != []:
                    result = script.run(script_data) # Execute the script with the collected data.

                    # Block Logic: Propagate the script's result to neighboring devices.
                    # Invariant: All neighbors receive the updated data for the given location.
                    for device in self.device.neighbourhood:
                        device.set_data(location, result)
                    
                    # Functional Utility: Update the current device's own data with the script's result.
                    self.device.set_data(location, result)

                self.device.map_locks[location].release() # Release the lock for the current location.

            # Block Logic: Synchronize all `DeviceThread` instances across all devices using the global barrier.
            # Invariant: All DeviceThread instances in the simulation will reach this barrier before any proceeds.
            self.device.barrier.wait()
            
            # Block Logic: Only the first worker thread (id == 0) performs cleanup for the next timepoint.
            # Invariant: `self.device.counter` is reset, and `self.device.timepoint_done` is cleared.
            if self.id == 0:
                self.device.counter = 0 # Reset script counter for the next timepoint.
                self.device.scripts = [] # Clear the scripts list for the next timepoint.
                self.device.timepoint_done.clear() # Clear the event for the next timepoint.

