

"""
@brief This module defines `Device` and `DeviceThread` classes for simulating a distributed system,
leveraging `ThreadPoolExecutor` for concurrent script execution.
@details It employs a `ReusableBarrierCond` for inter-device synchronization and per-location access locks
for thread-safe sensor data manipulation, enabling efficient and concurrent processing within the simulation.
"""

from threading import Event, Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import barrier # Assuming barrier.py contains ReusableBarrierCond

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It can receive and queue scripts for execution, which are then processed concurrently by a `DeviceThread`
    using a thread pool. Synchronization across devices is managed by a shared `ReusableBarrierCond`,
    and thread-safe access to per-location sensor data is ensured by `access_locks`.
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, utilizing a thread pool for parallel script execution
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

        self.device_barrier = None   # Reference to the shared ReusableBarrierCond for inter-device synchronization.
        self.script_received = Event() # Event to signal when new scripts are ready for execution.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's script assignment.

        self.scripts = []            # List to store assigned scripts and their locations.
        self.future_list = []        # List to hold futures of submitted tasks to the thread pool.
        self.access_locks = {}       # Dictionary to hold locks for each sensor data location.
        # Block Logic: Initialize a lock for each sensor data location for fine-grained access control.
        # Invariant: Each location in `sensor_data` has a corresponding Lock in `access_locks`.
        for location in sensor_data:
            self.access_locks[location] = Lock() # Create a new lock for each location.

        self.thread = DeviceThread(self) # The main worker thread for this device.
        self.thread.start()          # Start the device's execution thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared `ReusableBarrierCond` for all devices.
        @details This method ensures that a single `ReusableBarrierCond` instance is created by device 0
        and then distributed to all other devices in the simulation, allowing them to synchronize their
        execution across timepoints.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of the shared barrier.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `self.device_barrier` refers to a globally shared `ReusableBarrierCond` instance after setup.
        """
        if self.device_id == 0: # Only device 0 is responsible for initializing and distributing the barrier.
            device_barrier = barrier.ReusableBarrierCond(len(devices)) # Create a new reusable barrier.
            # Block Logic: Distribute the newly created barrier to all devices.
            # Invariant: All devices in `devices` receive a reference to the shared barrier.
            for device in devices:
                device.set_barrier(device_barrier)

    def set_barrier(self, device_barrier):
        """
        @brief Sets the shared `ReusableBarrierCond` instance for this device.
        @details This method is typically called by device 0 during setup to distribute
        the created barrier to all other devices.
        @param device_barrier (ReusableBarrierCond): The shared barrier instance.
        @block_logic Assigns the shared barrier to the device.
        @pre_condition `device_barrier` is an initialized `ReusableBarrierCond` object.
        @invariant `self.device_barrier` refers to the provided shared barrier.
        """
        self.device_barrier = device_barrier

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
        @brief Retrieves sensor data for a specific location, acquiring a lock for thread safety.
        @details This method acquires the `access_lock` for the specified `location` before
        reading the sensor data. The lock is *not* released here; it is expected to be
        released by the corresponding `set_data` call after the data has been processed.
        @param location (str): The location for which to retrieve data.
        @return object: The sensor data at the specified location, or None if the location is not found.
        @block_logic Provides thread-safe access to retrieve sensor data by acquiring a per-location lock.
        @pre_condition `location` exists as a key in `self.sensor_data` and `self.access_locks`.
        @invariant The `access_lock` for the `location` is acquired before returning the data.
        """
        if location in self.sensor_data:
            self.access_locks[location].acquire() # Acquire the lock for this specific location.
            result = self.sensor_data[location]   # Retrieve the data.
        else:
            result = None

        return result

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location and releases the corresponding lock.
        @details This method updates the internal sensor data for the specified `location` and then
        releases the `access_lock` that was acquired by a `get_data` call for the same location.
        @param location (str): The location whose data is to be updated.
        @param data (object): The new data value for the specified location.
        @block_logic Provides thread-safe access to update sensor data and releases the per-location lock.
        @pre_condition `location` exists as a key in `self.sensor_data` and `self.access_locks`.
                       The `access_lock` for this `location` must have been previously acquired.
        @invariant `self.sensor_data[location]` is updated, and the `access_lock` for `location` is released.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data   # Update the data.
            self.access_locks[location].release() # Release the lock for this specific location.

    def shutdown(self):
        """
        @brief Shuts down the device by joining its associated thread.
        @details This ensures that the device's worker thread completes its execution before the program exits.
        """
        self.thread.join()



class DeviceThread(Thread):
    """
    @brief The main worker thread for a Device instance.
    @details This thread orchestrates the device's operational cycle, including
    fetching neighbor information from the supervisor, managing the lifecycle of
    scripts submitted for concurrent execution via a `ThreadPoolExecutor`, and
    synchronizing with other `DeviceThread` instances using a shared barrier.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    abstracting multi-threaded script execution through a thread pool and ensuring proper
    coordination and data consistency within the distributed system.
    """

    def execute(self, neighbours, script, location):
        """
        @brief Executes a single script in a separate thread from the pool.
        @details This method collects data from neighboring devices and the parent device for
        the specified location, runs the given script with this collected data, and then
        propagates the results back to the neighbors and the parent device. It uses the
        parent device's `get_data` and `set_data` methods, which internally handle locks
        for thread safety.
        @param neighbours (list): A list of neighboring Device objects.
        @param script (object): The script object to be executed.
        @param location (str): The location relevant to the script and data processing.
        @block_logic Collects data, executes a script, and updates data across devices for a specific location.
        @pre_condition `neighbours` is a list of `Device` objects, `script` has a `run` method,
                       `location` is a valid data key. `device.access_locks` must be handled by `get_data` and `set_data`.
        @invariant `script_data` is populated, `script` is run, and relevant data is updated.
        """
        script_data = [] # List to accumulate data for the current script's execution.

        # Block Logic: Collect data from neighboring devices, excluding the current device itself.
        # Invariant: `script_data` contains data from neighbors for the given location.
        for device in neighbours:
            if device.device_id != self.device.device_id: # Avoid getting data from self again if already included.
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

            # Block Logic: Propagate the script's result to neighboring devices, excluding itself.
            # Invariant: All neighbors receive the updated data for the given location.
            for device in neighbours:
                if device.device_id != self.device.device_id: # Avoid setting data for self twice.
                    device.set_data(location, result)

            # Functional Utility: Update the current device's own data with the script's result.
            self.device.set_data(location, result)

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
        self.device = device # Reference to the parent Device object.
        self.thread_pool = ThreadPoolExecutor(max_workers=8) # Thread pool for concurrent script execution.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        @details This method continuously monitors the simulation state. For each timepoint,
        it retrieves neighbor information from the supervisor. If neighbors are available,
        it waits for script assignments to be complete (`timepoint_done`), then submits
        each assigned script to the `thread_pool` for concurrent execution. After all
        scripts are submitted, it waits for their completion (`concurrent.futures.wait`),
        resets the events, and finally synchronizes with other `DeviceThread` instances
        via the shared `device_barrier`. The loop terminates when the supervisor signals
        the end of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        parallel script execution via a thread pool, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `timepoint_done` and `script_received` events, `scripts` list, and `device_barrier`.
        @invariant The thread progresses through timepoints, processes scripts concurrently,
                   and ensures global synchronization. The thread pool is shut down gracefully on exit.
        """
        while True:
            # Functional Utility: Get information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # Block Logic: Check if the simulation should terminate.
            # Pre-condition: `neighbours` list indicates the current state of the network.
            # Invariant: The loop terminates if no neighbors are returned by the supervisor.
            if neighbours is None:
                break

            future_list = [] # List to hold Future objects returned by the thread pool.

            # Block Logic: Wait until script assignments for the current timepoint are complete.
            # Pre-condition: `self.device.timepoint_done` is an Event object.
            # Invariant: The thread proceeds only after the supervisor signals completion of script assignment.
            self.device.timepoint_done.wait()

            # Block Logic: Submit assigned scripts to the thread pool for concurrent execution.
            # Pre-condition: `self.device.script_received` is set if scripts are available.
            # Invariant: Each script is submitted as a task to the thread pool.
            if self.device.script_received.is_set():
                self.device.script_received.clear() # Clear the event after processing.

                # Block Logic: Iterate through assigned scripts and submit them to the thread pool.
                # Invariant: `future_list` contains Future objects for all submitted scripts.
                for (script, location) in self.device.scripts:
                    # Functional Utility: Submit the `execute` method to the thread pool.
                    future = self.thread_pool.submit(self.execute, neighbours, script, location)
                    future_list.append(future)

            # Functional Utility: Clear `timepoint_done` for the next timepoint.
            self.device.timepoint_done.clear()
            # Functional Utility: Set `script_received` for the next timepoint.
            self.device.script_received.set() # This seems like a bug or unconventional use, as script_received signals new scripts. It's set prematurely here.

            # Functional Utility: Clear the scripts list for the next timepoint.
            self.device.scripts = [] # Reset scripts list for the next timepoint.

            # Block Logic: Wait for all submitted tasks in the thread pool to complete.
            # Invariant: The thread does not proceed until all scripts have finished executing.
            concurrent.futures.wait(future_list)

            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances will reach this barrier before any proceeds to the next timepoint.
            self.device.device_barrier.wait()

        # Functional Utility: Shut down the thread pool gracefully upon loop termination.
        self.thread_pool.shutdown()

