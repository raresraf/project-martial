

"""
@brief This module defines `Device`, `DeviceThread`, and `ScriptThread` classes for simulating a distributed system.
@details It uses a `ReusableBarrierCond` for inter-device synchronization, employs a main device thread that
dispatches scripts to worker script threads in batches for concurrent processing, and utilizes per-location
locks for thread-safe sensor data access. This design aims to balance concurrency with controlled resource
management within a distributed sensor network simulation.
"""

from threading import Event, Thread, Lock
import barrier # Assuming barrier.py contains ReusableBarrierCond

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It can receive and queue scripts for execution, which are then processed by its dedicated
    `DeviceThread` by spawning `ScriptThread` instances in batches. Synchronization across devices
    is managed by a shared `ReusableBarrierCond`, and thread-safe access to per-location sensor data
    is ensured by `list_locks`.
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, utilizing a main thread that orchestrates batched parallel
    script execution and granular locking for data consistency and integrity.
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
        self.scripts_received = Event() # Event to signal when new scripts are ready for execution.
        self.scripts = []            # List to store assigned scripts and their locations.
        self.thread = DeviceThread(self) # The main worker thread for this device.
        self.data_lock = Lock()      # Lock for protecting `self.sensor_data` during access.
        self.list_locks = {}         # Dictionary to hold locks for each sensor data location (shared across devices).
        self.barrier = None          # Reference to the shared ReusableBarrierCond for inter-device synchronization.
        self.devices = None          # Reference to the list of all Device objects in the simulation.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives and distributes them among devices.
        @details This method initializes the global `ReusableBarrierCond` and `list_locks` only once
        by the first device (device 0) and then distributes them to all other devices in the simulation.
        It also starts the device's main `DeviceThread`.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of shared resources, and starting the main device thread.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `self.barrier` refers to a globally shared `ReusableBarrierCond` instance after setup.
                   `self.list_locks` is a shared dictionary of locks for sensor data locations.
                   The device's `DeviceThread` is started.
        """
        self.devices = devices # Store the list of all devices.

        # Block Logic: Initialize global barrier and map_locks only once by device 0.
        # Invariant: Global `barrier` and `list_locks` become initialized and shared.
        if self.device_id == self.devices[0].device_id: # Only device 0 performs initialization.
            self.barrier = barrier.ReusableBarrierCond(len(self.devices)) # Create a new reusable barrier.
            # Block Logic: Initialize `list_locks` with a lock for each unique sensor data location across all devices.
            # Invariant: `self.list_locks` contains a Lock object for every unique sensor location.
            for dev in self.devices:
                for location in dev.sensor_data:
                    self.list_locks[location] = Lock() # Create a new lock for each location.
        else:
            # Other devices retrieve the shared barrier and list_locks from device 0.
            self.barrier = devices[0].get_barrier()
            self.list_locks = devices[0].get_list_locks()
        
        self.thread.start() # Start the device's main execution thread.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location.
        @details If a script is provided, it's appended to the device's script queue.
        The `scripts_received` event is always set to signal the `DeviceThread` that
        new scripts are available for processing (or that the current batch is complete).
        @param script (object): The script object to be executed, or None to signal the end of a batch.
        @param location (str): The location associated with the script or data.
        @block_logic Handles the assignment of new scripts and signals readiness for execution.
        @pre_condition `self.scripts` is a list, `self.scripts_received` is an Event object.
        @invariant If `script` is not None, it's added to `self.scripts`. `scripts_received` is always set.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the queue.
        else:
            self.scripts_received.set() # Signal that all scripts for the current batch have been assigned.

    def get_barrier(self):
        """
        @brief Returns the shared `ReusableBarrierCond` instance.
        @details This method allows other devices to retrieve the barrier instance created by device 0.
        @return ReusableBarrierCond: The shared barrier instance.
        @pre_condition The barrier must have been initialized by device 0.
        """
        return self.barrier

    def get_list_locks(self):
        """
        @brief Returns the shared dictionary of `list_locks`.
        @details This method allows other devices to retrieve the `list_locks` dictionary
        created by device 0, ensuring consistent access control to sensor data locations.
        @return dict: A dictionary where keys are locations and values are `Lock` objects.
        @pre_condition `list_locks` must have been initialized by device 0.
        """
        return self.list_locks

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location in a thread-safe manner.
        @details This method acquires `data_lock` before accessing `sensor_data` to ensure
        thread safety during reads.
        @param location (str): The location for which to retrieve data.
        @return object: The sensor data at the specified location, or None if the location is not found.
        @block_logic Provides thread-safe access to retrieve sensor data.
        @pre_condition `self.sensor_data` is a dictionary, `self.data_lock` is an initialized Lock.
        @invariant Returns data associated with `location` if present, otherwise None, under lock protection.
        """
        with self.data_lock: # Acquire lock to ensure thread-safe reading of sensor data.
            if location in self.sensor_data:
                data = self.sensor_data[location]
            else:
                data = None
        return data

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location in a thread-safe manner.
        @details This method acquires `data_lock` before modifying `sensor_data` to ensure
        thread safety during writes.
        @param location (str): The location whose data is to be updated.
        @param data (object): The new data value for the specified location.
        @block_logic Provides thread-safe access to update sensor data.
        @pre_condition `self.sensor_data` is a dictionary, `self.data_lock` is an initialized Lock.
        @invariant If `location` is a key in `self.sensor_data`, its value is updated under lock protection.
        """
        with self.data_lock: # Acquire lock to ensure thread-safe modification of sensor data.
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
    @details This thread orchestrates the device's operational cycle, including
    fetching neighbor information from the supervisor, waiting for script assignments,
    and then processing these scripts by spawning `ScriptThread` instances in batches
    for concurrent execution. It ensures inter-device synchronization using a shared `ReusableBarrierCond`.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    abstracting multi-threaded script execution through batched worker threads and coordinating
    with the global barrier to ensure proper progression of the distributed system.
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
        it waits until `scripts_received` is set (signaling that script assignments for the
        current batch are complete), then processes the assigned scripts by creating and
        starting `ScriptThread` instances in batches of 8. After all `ScriptThread` instances
        complete, it clears `scripts_received` and synchronizes with other `DeviceThread`
        instances via the shared `ReusableBarrierCond`. The loop terminates when the supervisor
        signals the end of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        batched parallel script execution, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `scripts` list, `scripts_received` event, `barrier`, and `list_locks`.
        @invariant The thread progresses through timepoints, processes scripts concurrently in batches,
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

            # Block Logic: Wait until all scripts for the current batch have been assigned.
            # Pre-condition: `self.device.scripts_received` is an Event object.
            # Invariant: The thread proceeds only after the supervisor signals completion of script assignment.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear() # Clear the event for the next batch.

            threads = [] # List to hold ScriptThread instances for batched execution.
            # Block Logic: Iterate through assigned scripts, creating and managing ScriptThread instances in batches.
            # Invariant: Scripts are processed concurrently in batches, and `threads` list holds active ScriptThreads.
            for (script, location) in self.device.scripts:
                threads.append(
                    ScriptThread(self.device, script, location, neighbours))
                # Block Logic: If 8 threads are collected, start them and wait for completion before proceeding.
                # Invariant: ScriptThreads are executed in batches to manage concurrency.
                if len(threads) == 8: # Batch size of 8.
                    for thr in threads:
                        thr.start()
                    for thr in threads:
                        thr.join()
                    threads = [] # Reset the batch.
            
            # Block Logic: Handle any remaining threads that didn't form a full batch of 8.
            # Invariant: All remaining ScriptThreads are started and joined.
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()

            # Functional Utility: Clear the scripts list for the next timepoint.
            self.device.scripts = [] # Reset scripts list for the next timepoint.
            
            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances will reach this barrier before any proceeds to the next timepoint.
            self.device.barrier.wait()


class ScriptThread(Thread):
    """
    @brief A worker thread dedicated to executing a single assigned script for a Device instance.
    @details This thread processes a specific script for a given location, collects data from the
    parent device and its neighbors, executes the script's logic, and updates sensor data
    in a thread-safe manner using a location-specific lock.
    @architectural_intent Enhances parallelism by allowing multiple scripts to run concurrently,
    with controlled resource access to prevent race conditions during data manipulation.
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
        @details This method acquires a location-specific lock to control access to shared data.
        It collects data from the parent device and its neighbors for the specified location,
        executes the assigned script, and then updates the relevant sensor data for the
        device and its neighbors. Finally, it releases the location-specific lock.
        @block_logic Processes a single script for a specific location, ensuring thread-safe data access.
        @pre_condition `self.script` is an object with a `run` method, `self.device.list_locks`
                       contains a Lock for `self.location`.
        @invariant The script is executed, and data is updated under the protection of a location lock.
        """
        # Block Logic: Acquire a lock for the specific location to ensure exclusive access to its data.
        # Invariant: Only one ScriptThread can modify or read data for `self.location` at a time.
        self.device.list_locks[self.location].acquire()

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

        self.device.list_locks[self.location].release() # Release the lock for the current location.

