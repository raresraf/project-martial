"""
@file device.py
@brief Implements a simulated computational device with multi-threaded script execution for distributed systems.

This module defines the `Device` class, representing a node in a distributed system,
`DeviceThread` for orchestrating script execution on that device, and `Worker`
threads for parallel processing of scripts. It includes mechanisms for inter-device
synchronization using barriers and per-location shared locks for data consistency.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem # Assumed to be a custom barrier implementation


class Device(object):
    """
    @class Device
    @brief Represents a simulated computational device in a distributed sensing and processing network.

    Each `Device` instance can hold sensor data, process assigned scripts,
    and interact with a `Supervisor` to coordinate with other devices.
    It manages its own thread of execution (`DeviceThread`) and uses synchronization
    primitives (barriers and shared locks) for coordinated data access and script execution.

    @attribute device_id (int): A unique identifier for the device.
    @attribute sensor_data (dict): A dictionary holding sensor data, keyed by location.
    @attribute supervisor (Supervisor): A reference to the central supervisor managing all devices.
    @attribute scripts (list): A list of (script, location) tuples to be executed in the current timepoint.
    @attribute devices (list): A list of all `Device` objects in the system (populated by supervisor or device 0).
    @attribute cores (int): The number of worker threads (cores) this device can simulate.
    @attribute barrier (ReusableBarrierSem): A shared barrier for inter-device synchronization across timepoints.
    @attribute shared_locks (list): A list of locks, one for each possible sensor data location, to protect access.
    @attribute timepoint_done (Event): Synchronization event, set when a timepoint's script assignment is complete.
    @attribute thread (DeviceThread): The dedicated control thread for this device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Sets up the device's unique ID, initial sensor data, supervisor reference,
        and various synchronization primitives and data structures required
        for its operation in the distributed system. It also starts the
        device's dedicated `DeviceThread`.

        @param device_id (int): A unique identifier for this device.
        @param sensor_data (dict): Initial sensor data for this device, typically a dictionary mapping locations to data values.
        @param supervisor (Supervisor): The supervisor object responsible for managing this device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        self.devices = []
        self.cores = 8 # Inline: Represents the number of worker threads for parallel script execution.
        self.barrier = None
        self.shared_locks = [] # Inline: List to hold locks for each data location.
        self.timepoint_done = Event() # Inline: Event to signal completion of script assignment for a timepoint.
        self.thread = DeviceThread(self) # Inline: Creates and starts the device's control thread.
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        This method provides a human-readable string that identifies the device
        by its `device_id`.

        @return (str): A string in the format "Device [device_id]".
        """
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """
        @brief Sets the shared reusable barrier for inter-device synchronization.

        @param barrier (ReusableBarrierSem): The barrier synchronization object.
        """
        self.barrier = barrier

    def set_locks(self, locks):
        """
        @brief Sets the shared list of location-specific locks.

        @param locks (list): A list of `threading.Lock` objects for each data location.
        """
        self.shared_locks = locks

    def setup_devices(self, devices):
        """
        @brief Configures global synchronization primitives and shared resources.

        This method is primarily called by the supervisor or device 0 to
        initialize the shared barrier for all devices and ensure all devices
        have access to a consistent set of location-specific locks.

        @param devices (list): A list of all `Device` objects in the simulated system.
        """
        self.devices = devices

        # Block Logic: Device 0 initializes the global reusable barrier for all devices.
        if self.device_id == 0:
            lbarrier = ReusableBarrierSem(len(devices))
            # Block Logic: Distributes the initialized barrier to all devices.
            for dev in devices:
                dev.set_barrier(lbarrier)

        # Block Logic: Initializes shared locks for data locations across all devices.
        # This ensures all devices use the same lock for a given location.
        max_loc = 0
        if self.sensor_data: # Prevent error if sensor_data is empty
            max_loc = max(self.sensor_data.keys(), key=int)

        # Block Logic: Creates location-specific locks if the current set is insufficient for max_loc.
        if  max_loc + 1 > len(self.shared_locks):
            llocks = []
            # Block Logic: Initializes a Lock for each potential data location.
            for _ in range(max_loc + 1):
                llocks.append(Lock())
            self.set_locks(llocks) # Inline: Sets the device's shared_locks.
            # Block Logic: Distributes the newly created shared locks to all devices.
            for dev in self.devices:
                dev.set_locks(llocks)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        This method adds a new (script, location) tuple to the device's
        `scripts` queue. If `script` is None, it sets the `timepoint_done`
        event, signaling that all scripts for the current timepoint have been assigned.

        @param script (object): The script object to be executed.
        @param location (int): The sensor data location that the script pertains to.
        """
        if script is not None:
            self.scripts.append((script, location)) # Inline: Adds the script and location to the list of pending scripts.
        else:
            self.timepoint_done.set() # Inline: Signals that all scripts for the current timepoint are assigned.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        This method provides access to the sensor data stored on the device
        for a given `location`. It does not explicitly use locks here,
        relying on the caller (Worker) to acquire the appropriate `shared_locks`.

        @param location (int): The specific location for which to retrieve data.
        @return (any): The data at the specified location, or `None` if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location] # Inline: Returns the data if the location exists.
        else:
            return None # Inline: Returns None if the location is not found.

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.

        This method updates the `sensor_data` at a given `location` with the
        new `data`. It does not explicitly use locks here, relying on the caller
        (Worker) to acquire the appropriate `shared_locks` for thread safety.

        @param location (int): The specific location for which to set data.
        @param data (any): The new data value to be stored at the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data # Inline: Updates the sensor data at the specified location.

    def shutdown(self):
        """
        @brief Shuts down the device's main thread.

        This method ensures a clean termination by joining the device's
        dedicated control thread (`DeviceThread`).
        """
        self.thread.join() # Inline: Waits for the `DeviceThread` to complete its execution.


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief Manages the asynchronous execution of scripts on a `Device` using multiple worker threads.

    This class extends `threading.Thread` to provide a dedicated control thread
    for each `Device` instance. It orchestrates the distribution of scripts
    to available `Worker` threads, manages inter-device synchronization at
    timepoint boundaries, and oversees the overall simulation flow for the device.

    @attribute device (Device): A reference to the `Device` object this thread is managing.
    """

    def __init__(self, device):
        """
        @brief Initializes a new `DeviceThread` instance.

        Sets up the thread with a descriptive name and stores a reference
        to the `Device` object it will manage.

        @param device (Device): The `Device` instance that this thread will operate on.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def distribute_scripts(self, scripts):
        """
        @brief Distributes assigned scripts among the device's worker threads.

        This method takes a list of scripts and assigns them to worker-specific
        lists in a round-robin fashion, balancing the load across available cores.

        @param scripts (list): A list of (script, location) tuples to be executed.
        @return (list): A list of lists, where each inner list contains scripts for a specific worker.
        """
        worker_scripts = []
        # Block Logic: Initializes a list for each worker to store its assigned scripts.
        for _ in range(self.device.cores):
            worker_scripts.append([])
        i = 0
        # Block Logic: Distributes scripts to workers in a round-robin manner.
        for script in scripts:
            worker_scripts[i % self.device.cores].append(script)
            i = i + 1
        return worker_scripts

    def run(self):
        """
        @brief The main execution loop for the device's control thread.

        This method continuously executes in a loop, simulating the device's
        operation across multiple timepoints. It performs the following key functions:
        - Retrieves neighbor information from the supervisor.
        - Waits for all scripts for the current timepoint to be assigned.
        - Distributes scripts to worker threads for parallel execution.
        - Manages worker threads' lifecycle (start, join).
        - Synchronizes with other devices at timepoint ends using a shared barrier.
        """
        # Block Logic: Main simulation loop for the device's control thread.
        # Invariant: Each iteration manages script distribution and synchronization for a timepoint.
        while True:
            # Block Logic: Retrieves the current list of neighboring devices from the supervisor.
            # This list is dynamic and may change based on simulation state.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks if the simulation should terminate (no neighbors means end).
            # Precondition: If `neighbours` is None, it indicates the end of the simulation.
            if neighbours is None:
                break # Inline: Exits the main simulation loop.

            # Block Logic: Waits for all scripts for the current timepoint to be assigned.
            # This is signaled by `Device.assign_script(None, None)`.
            self.device.timepoint_done.wait()

            # Block Logic: Creates and manages worker threads for parallel script execution.
            inner_workers = [] # Inline: List to hold worker thread objects.

            # Block Logic: Distributes the collected scripts among the available worker threads.
            worker_scripts = self.distribute_scripts(self.device.scripts)
            # Block Logic: Creates and starts a `Worker` thread for each set of assigned scripts.
            for worker_scr in worker_scripts:
                inner_thread = Worker(worker_scr, neighbours, self.device) # Inline: Instantiates a new Worker.
                inner_workers.append(inner_thread) # Inline: Adds the worker to the list.
                inner_thread.start() # Inline: Starts the worker thread.

            # Block Logic: Waits for all worker threads to complete their assigned scripts.
            for thr in inner_workers:
                thr.join() # Inline: Blocks until the worker thread finishes.

            # Inline: Clears the `scripts` list for the next timepoint.
            self.device.scripts = []
            # Inline: Clears the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes all devices at the end of the current timepoint.
            # All devices wait here until every device has finished its timepoint processing.
            self.device.barrier.wait()


class Worker(Thread):
    """
    @class Worker
    @brief A dedicated thread for executing a subset of scripts on a `Device`.

    Each `Worker` operates under the direction of its parent `DeviceThread`,
    performing script execution for a given set of scripts. It collects data
    from neighboring devices and its own device, executes the script, and
    updates sensor data while ensuring data consistency using shared locks.

    @attribute script_loc (list): A list of (script, location) tuples assigned to this worker.
    @attribute neighbours (list): A list of neighboring `Device` objects for data exchange.
    @attribute script_data (list): Temporary list to hold collected sensor data for script execution.
    @attribute device (Device): A reference to the parent `Device` object.
    """

    def __init__(self, script_loc, neighbours, device):
        """
        @brief Initializes a new `Worker` thread instance.

        Sets up the worker with its assigned scripts, references to neighbors,
        and its parent device.

        @param script_loc (list): A list of (script, location) tuples for this worker to execute.
        @param neighbours (list): A list of neighboring `Device` objects.
        @param device (Device): The `Device` instance that this worker belongs to.
        """
        Thread.__init__(self)
        self.script_loc = script_loc
        self.neighbours = neighbours
        self.script_data = [] # Inline: Temporary storage for data collected for a script.
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the Worker thread.

        This method iterates through the scripts assigned to it. For each script,
        it acquires a shared lock for the corresponding data location, collects
        data from neighboring devices and its own device, executes the script,
        updates the relevant sensor data, and then releases the lock.
        """
        # Block Logic: Iterates through each (script, location) pair assigned to this worker.
        for (script, location) in self.script_loc:
            # Block Logic: Acquires a shared lock for the specific sensor data location.
            # This ensures exclusive access to the data at this location during script execution.
            self.device.shared_locks[location].acquire()
            self.script_data = [] # Inline: Resets collected data for the new script.

            # Block Logic: Collects sensor data from neighboring devices for the current location.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    self.script_data.append(data)

            # Block Logic: Collects sensor data from the current device itself for the current location.
            data = self.device.get_data(location)
            if data is not None:
                self.script_data.append(data)

            # Block Logic: Executes the script if any data was collected.
            if self.script_data != []:
                result = script.run(self.script_data) # Inline: Executes the script with collected data.

                # Block Logic: Updates sensor data on neighboring devices with the script's result.
                for dev in self.neighbours:
                    dev.set_data(location, result)

                # Block Logic: Updates sensor data on its own device with the script's result.
                self.device.set_data(location, result)

            # Block Logic: Releases the shared lock for the data location.
            self.device.shared_locks[location].release()
