"""
@brief This module defines `Device`, `DeviceThread`, and `ScriptWorkerThread` classes for simulating a distributed system.
@details It uses a `ReusableBarrier` for inter-device synchronization, employs a main device thread that dispatches
scripts to worker threads for concurrent processing, and utilizes a `BoundedSemaphore` to limit concurrency
and class-level location-specific locks for thread-safe sensor data access. This design aims to optimize
concurrent data processing and ensure data integrity within a distributed sensor network simulation.
"""

from threading import Event, Thread, Lock, BoundedSemaphore
from barrier import ReusableBarrier


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It can receive and queue scripts for execution, which are then processed by its dedicated
    `DeviceThread` by spawning `ScriptWorkerThread` instances for concurrent processing.
    Synchronization across devices is managed by a shared class-level `ReusableBarrier` (`timepoint_barrier`),
    and a `BoundedSemaphore` (`max_threads_semaphore`) is used to limit the number of concurrently
    executing `ScriptWorkerThread` instances.
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, utilizing worker threads for parallel script execution
    and granular locking for data consistency, with controlled concurrency.
    """
    
    timepoint_barrier = None # Class-level shared ReusableBarrier for inter-device synchronization across timepoints.
    barrier_lock = Lock()    # Lock to protect the initialization of `timepoint_barrier`.

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

        # BoundedSemaphore to limit the number of concurrently running ScriptWorkerThread instances.
        self.max_threads_semaphore = BoundedSemaphore(8) # Limits concurrent script execution to 8 threads.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    @staticmethod
    def setup_devices(devices):
        """
        @brief Sets up the class-level shared `ReusableBarrier` (`timepoint_barrier`) for all devices.
        @details This static method ensures that a single `ReusableBarrier` instance is created and
        initialized only once across all `Device` instances using a double-checked locking pattern
        with `barrier_lock`. This barrier is then used by all devices for timepoint synchronization.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized, thread-safe initialization of the global timepoint barrier.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `Device.timepoint_barrier` is an initialized `ReusableBarrier` instance.
        """
        # Block Logic: Double-checked locking to ensure `Device.timepoint_barrier` is initialized only once.
        if Device.timepoint_barrier is None: # First check without lock for performance.
            Device.barrier_lock.acquire() # Acquire lock for thread-safe initialization.
            if Device.timepoint_barrier is None: # Second check inside lock.
                # Functional Utility: Create a new reusable barrier, sized for all devices.
                Device.timepoint_barrier = ReusableBarrier(len(devices))
            Device.barrier_lock.release() # Release lock.

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
        It's assumed that external synchronization (e.g., through `ScriptWorkerThread`'s locks)
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



class DeviceThread(Thread):
    """
    @brief The main worker thread for a Device instance.
    @details This thread orchestrates the device's operational cycle for each timepoint.
    It fetches neighbor information, waits for script assignments, and then dispatches
    these scripts to individual `ScriptWorkerThread` instances for concurrent execution.
    It manages the concurrency of script workers using `max_threads_semaphore` and
    synchronizes with other `DeviceThread` instances using the class-level `timepoint_barrier`.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    abstracting multi-threaded script execution through worker threads and ensuring proper
    coordination and data consistency within the distributed system.
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
        then processes the assigned scripts by creating and starting `ScriptWorkerThread` instances.
        Each `ScriptWorkerThread` acquires a permit from `max_threads_semaphore` before starting,
        limiting concurrent script execution. After all script worker threads complete, it clears
        `timepoint_done` and synchronizes with other `DeviceThread` instances via the global
        `Device.timepoint_barrier`. The loop terminates when the supervisor signals the end
        of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        parallel script execution with bounded concurrency, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `scripts` list, `timepoint_done` event, `max_threads_semaphore`, and `Device.timepoint_barrier`.
        @invariant The thread progresses through timepoints, processes scripts concurrently (up to a limit),
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

            threads = [] # List to hold ScriptWorkerThread instances.
            # Block Logic: Iterate through assigned scripts, creating and managing ScriptWorkerThread instances.
            # Invariant: Each script leads to the creation and start of a `ScriptWorkerThread`.
            for (script, location) in self.device.scripts:
                self.device.max_threads_semaphore.acquire() # Acquire a permit before starting a new worker.

                # Functional Utility: Create a new ScriptWorkerThread for each script.
                worker_thread = ScriptWorkerThread(self.device, neighbours, location, script)
                threads.append(worker_thread) # Add the thread to the list for joining later.
                worker_thread.start() # Start the ScriptWorkerThread to execute the script concurrently.

            # Block Logic: Wait for all spawned ScriptWorkerThread instances to complete their execution.
            # Invariant: The DeviceThread will not proceed until all its ScriptWorkerThread children have finished.
            for thread in threads:
                thread.join()

            # Functional Utility: Clear the scripts list for the next timepoint.
            self.device.scripts = [] # Reset scripts list for the next timepoint.
            # Functional Utility: Clear the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()

            # Block Logic: Synchronize with other DeviceThread instances via the shared class-level barrier.
            # Invariant: All DeviceThread instances in the simulation will reach this barrier before any proceeds to the next timepoint.
            Device.timepoint_barrier.wait()


class ScriptWorkerThread(Thread):
    """
    @brief A worker thread dedicated to executing a single assigned script for a Device instance.
    @details This thread processes a specific script for a given location, collects data from the
    parent device and its neighbors, executes the script's logic, and updates sensor data
    in a thread-safe manner using class-level location-specific locks. After completing its task,
    it releases the permit from the parent device's `max_threads_semaphore`.
    @architectural_intent Enhances parallelism by allowing multiple scripts to run concurrently,
    with controlled resource access through both a `BoundedSemaphore` (from `DeviceThread`) and
    location-specific locks (`locations_lock`) to prevent race conditions during data manipulation.
    """
    
    locations_lock = {} # Class-level dictionary to hold locks for each sensor data location (shared across all ScriptWorkerThreads).

    def __init__(self, device, neighbours, location, script):
        """
        @brief Initializes a new ScriptWorkerThread instance.
        @param device (Device): The parent Device object that this script thread serves.
        @param neighbours (list): A list of neighboring Device objects from which to collect sensor data.
        @param location (str): The location associated with the script for which data is processed.
        @param script (object): The script object to be executed.
        """
        super(ScriptWorkerThread, self).__init__() # Initialize the base Thread class.
        self.device = device # Reference to the parent Device object.
        self.neighbours = neighbours # List of neighboring devices.
        self.location = location # The sensor data location this script pertains to.
        self.script = script # The script to execute.

        # Block Logic: Ensure a lock exists for the given location, initializing it if not present.
        # Invariant: `ScriptWorkerThread.locations_lock[location]` holds a valid Lock object.
        if location not in ScriptWorkerThread.locations_lock:
            ScriptWorkerThread.locations_lock[location] = Lock() # Create a new lock for this location if it doesn't exist.

    def run(self):
        """
        @brief The main execution logic for the ScriptWorkerThread.
        @details This method acquires a location-specific lock from the class-level `locations_lock`
        to control exclusive access to data at its assigned location. It collects data from the
        parent device and its neighbors, executes the assigned script, and then updates the
        relevant sensor data for the device and its neighbors. Finally, it releases the
        location-specific lock and the permit from the parent device's `max_threads_semaphore`.
        @block_logic Processes a single script for a specific location, ensuring thread-safe data access.
        @pre_condition `self.script` is an object with a `run` method, `ScriptWorkerThread.locations_lock`
                       contains a Lock for `self.location`. `self.device.max_threads_semaphore` has been acquired.
        @invariant The script is executed, data is updated under the protection of a location lock,
                   and the semaphore permit is released.
        """
        # Block Logic: Acquire a lock for the specific location to ensure exclusive access to its data.
        # Invariant: Only one ScriptWorkerThread can modify or read data for `self.location` at a time.
        ScriptWorkerThread.locations_lock[self.location].acquire()

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

        ScriptWorkerThread.locations_lock[self.location].release() # Release the lock for the current location.

        # Functional Utility: Release the permit from the parent device's semaphore.
        self.device.max_threads_semaphore.release() # This must be called to allow other workers to start.
