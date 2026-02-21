"""
@file device.py
@brief Implements a simulated device and its threading model for distributed sensing and script execution.

This module defines the `Device` and `DeviceThread` classes, which together
simulate a distributed system of interconnected devices. Each `Device` manages
its local sensor data, communicates with a supervisor, and executes assigned scripts.
The `DeviceThread` class enables concurrent processing within each device,
handling neighborhood discovery, script execution, and synchronization.

Architecture:
- `Device`: Represents a single node in a distributed sensing network.
  Manages local state, inter-device communication mechanisms (via supervisor),
  and script assignments.
- `DeviceThread`: A worker thread within a `Device` responsible for
  concurrent execution of tasks, including neighborhood updates and script processing.
- `cond_barrier.ReusableBarrier`: Used for synchronization across multiple threads
  within a device and across multiple devices at timepoint boundaries.

"""
@8daf585f-de8c-45ba-9863-7279dde65b42/device.py
@brief Implements Device and DeviceThread for distributed simulation, with advanced multi-threading and synchronization.

This module defines the core components for a multi-threaded simulated device environment.
Each `Device` manages its state and interactions, delegating script execution to multiple
`DeviceThread` instances. Synchronization is critical and is handled through a combination
of local per-device barriers, global barriers (managed by `cond_barrier.ReusableBarrier`),
and per-location locks to ensure data consistency in a concurrent processing model.
"""

import cond_barrier
from threading import Event, Thread, Lock


class Device(object):
    """
    @brief Represents a simulated device with multi-threaded processing capabilities.

    Each Device instance is responsible for managing its sensor data, interacting
    with a supervisor for network topology, and orchestrating the execution of
    multiple `DeviceThread` instances. It uses a combination of global and local
    barriers (from `cond_barrier`) and per-location locks to ensure synchronized
    and consistent data processing across the distributed simulation.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance, setting up its state and synchronization components.

        Initializes device-specific attributes such as ID, sensor data, and supervisor reference.
        Crucially, it sets up multiple synchronization primitives:
        - `script_received`: Event to signal new script assignments.
        - `timepoint_done`: Event to signal completion of a timepoint's processing.
        - `threads`: List to hold multiple `DeviceThread` instances for parallel processing.
        - `neighbourhood`: Stores references to neighboring devices.
        - `map_locks`: Dictionary for per-location data access control (locks).
        - `threads_barrier`: Barrier for internal synchronization of this device's multiple threads.
        - `barrier`: Reference to a global barrier for inter-device synchronization.
        - `counter`: Tracks script processing progress within the device.
        - `threads_lock`: Lock for controlling access to shared resources among this device's threads.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
        @param supervisor: A reference to the supervisor object managing the device network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.threads = []

        self.neighbourhood = None
        self.map_locks = {}
        self.threads_barrier = None
        self.barrier = None
        self.counter = 0
        self.threads_lock = Lock()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the device's internal threads and coordinates with a global barrier.

        If this is the device with ID 0, it initializes the global `cond_barrier.ReusableBarrier`
        which will coordinate all devices (num_threads * 8, assuming 8 threads per device).
        It then propagates this global barrier and shared `map_locks` to all other devices.
        Finally, it creates and starts 8 `DeviceThread` instances for itself, each with a
        reference to a local `threads_barrier` for internal synchronization.

        @param devices: A list of all Device instances in the simulation.
        """
        
        if self.device_id == 0:
            num_threads = len(devices)

            # Invariant: The barrier is initialized with a count equal to the total
            # number of device threads across all devices in the simulation.
            self.barrier = cond_barrier.ReusableBarrier(num_threads * 8)

            # Block Logic: Propagates the shared barrier and map locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.map_locks = self.map_locks

        # Block Logic: Initializes internal threads and their synchronization barrier
        # for concurrent operations within this specific device.
        self.threads_barrier = cond_barrier.ReusableBarrier(8)
        for i in range(8):
            self.threads.append(DeviceThread(self, i, self.threads_barrier))

        # Block Logic: Starts all worker threads associated with this device.
        for thread in self.threads:
            thread.start()


    def assign_script(self, script, location):
        """
        @brief Assigns a script for execution and manages related synchronization and locking.

        If a script is provided, it's appended to the `scripts` list, and `script_received`
        event is set. If `script` is None, it signals the completion of script assignments
        for the current timepoint by setting `timepoint_done`.
        Crucially, if a lock for the given `location` does not exist in `map_locks`,
        a new `Lock` is created and associated with that location.

        @param script: The script object to be executed, or None to signal end of assignments.
        @param location: The data location (e.g., sensor ID) the script operates on.
        """
        

        If a script is provided, it's added to the device's script queue,
        and an event is set to signal script availability to worker threads.
        If no script is provided (None), it signals that the current timepoint
        processing is complete for script assignment.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """

        # Block Logic: Manages script assignment and signals to worker threads.
        # Pre-condition: 'script' can be an executable object or None.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

        # Block Logic: Initializes a lock for a given data location if one doesn't already exist.
        # This ensures exclusive access to sensor data at specific locations during script execution.
        if location not in self.map_locks:
            self.map_locks[location] = Lock()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The key identifying the sensor data to retrieve.
        @return: The sensor data at the specified location, or None if not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.

        Updates the sensor data if the location already exists in the device's
        sensor data dictionary.

        @param location: The key identifying the sensor data to update.
        @param data: The new data value to set for the specified location.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down all processing threads associated with this device.

        This method iterates through all `DeviceThread` instances belonging to this
        device and waits for each to complete its execution, ensuring a clean
        and orderly shutdown of all concurrent activities.
        """
        

        Functional Utility: Ensures proper termination and cleanup of resources
        associated with the device's concurrent execution.
        """

        # Block Logic: Waits for all internal worker threads to complete their execution.
        # This is a critical step for graceful shutdown, preventing resource leaks or unexpected termination.
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    @brief Represents an individual worker thread responsible for processing scripts within a Device.

    Each `DeviceThread` operates as part of a pool of threads managed by a `Device` instance.
    It participates in local device-level synchronization as well as global simulation-level
    synchronization. Its primary task is to fetch scripts, gather necessary data,
    execute scripts, and disseminate results, all while adhering to concurrency controls.
    """
    

    def __init__(self, device, id, barrier):
        """
        @brief Initializes a new DeviceThread instance.

        Sets up the thread with a descriptive name, associates it with
        its parent `Device` instance, assigns a unique ID within the device's
        thread pool, and provides a reference to the local thread barrier.

        @param device: The parent Device instance this thread belongs to.
        @param id: A unique identifier for this thread within its device's thread pool.
        @param barrier: The `cond_barrier.ReusableBarrier` for local synchronization among device threads.
        """
        

        @param device: A reference to the parent Device.
        @param id: A unique identifier for this thread within the device.
        @param barrier: A reusable barrier for synchronizing threads within the device.
        """

        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id = id
        self.thread_barrier = barrier

    def run(self):
        """
        @brief The main execution loop for an individual DeviceThread.

        This loop continuously performs the following steps:
        1.  (Only for thread ID 0) Fetches the current set of neighboring devices from the supervisor
            to update the device's awareness of its immediate network.
        2.  Synchronizes all threads within the current device using a local barrier, ensuring
            all internal threads are ready before proceeding to process a new timepoint.
        3.  Terminates if no neighbors are found, signifying the end of the simulation for this device.
        4.  Waits for the supervisor to signal that a new timepoint has begun and scripts are ready.
        5.  Enters a loop to process assigned scripts:
            a.  Acquires a lock to safely get the next script to process, ensuring each script
                is handled exactly once by one thread.
            b.  Acquires a per-location lock to prevent race conditions when accessing/modifying data
                for a specific sensor location.
            c.  Collects relevant data from neighboring devices and the device's own sensors for the
                current script's target location.
            d.  Executes the script with the collected data.
            e.  Disseminates the processed result by updating the sensor data of neighboring devices
                and the device itself.
            f.  Releases the per-location lock.
        6.  Waits at a global barrier, synchronizing with all other devices (and their threads)
            in the simulation, ensuring all devices have completed their timepoint processing.
        7.  (Only for thread ID 0) Resets the device's internal state (script counter and timepoint event)
            to prepare for the next simulation timepoint.
        """
        while True:
            # Block Logic: The first thread (ID 0) is responsible for updating the device's neighborhood information.
            # Functional Utility: Centralizes network topology updates to avoid redundant calls and ensure consistency.
            if self.id == 0:
                self.device.neighbourhood = self.device.supervisor.get_neighbours()

            # Block Logic: Synchronizes all internal threads of the current device.
            # Functional Utility: Ensures that all worker threads within this device are aligned before
            #                      advancing to processing timepoint data.
            self.thread_barrier.wait()

            # Invariant: If the device has no more neighbors, it signals the end of its simulation lifecycle.
            if self.device.neighbourhood is None:
                break

            # Block Logic: Waits for the supervisor to signal the start of a new timepoint and availability of scripts.
            # Functional Utility: Orchestrates the progression of simulation timepoints, ensuring data consistency.
            self.device.timepoint_done.wait()

            # Block Logic: Iteratively processes all assigned scripts for the current timepoint.
            # Invariant: Scripts are processed one by one, with proper locking for data integrity.
            while True:
                # Block Logic: Critical section for safely acquiring the next script to process.
                # Functional Utility: Ensures that multiple threads within the same device do not
                #                      process the same script simultaneously and all scripts are covered.
                with self.device.threads_lock:
                    # Invariant: Breaks the loop once all scripts assigned for the timepoint have been processed.
                    if self.device.counter == len(self.device.scripts):
                        break
                    (script, location) = self.device.scripts[self.device.counter]
                    self.device.counter = self.device.counter + 1
                
                # Block Logic: Acquires a lock specific to the data location (e.g., sensor ID).
                # Functional Utility: Prevents race conditions and ensures atomic updates
                #                      when multiple threads might try to modify data at the same location.
                self.device.map_locks[location].acquire()
                script_data = []

                # Block Logic: Gathers relevant sensor data from neighboring devices for the current script's location.
                # Functional Utility: Collects necessary input for the script based on the current network state.
                for device in self.device.neighbourhood:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Block Logic: Includes the device's own sensor data for the current script's location.
                # Functional Utility: Ensures the script considers the device's local state.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if there is any data to process.
                if script_data != []:
                    # Block Logic: Executes the assigned script with the aggregated data.
                    # Architectural Intent: Decouples computational logic from data management,
                    #                      allowing dynamic script execution based on current data.
                    result = script.run(script_data)

                    # Block Logic: Disseminates the computed result to neighboring devices.
                    # Functional Utility: Propagates state changes across the network as a result of script execution.
                    for device in self.device.neighbourhood:
                        device.set_data(location, result)
                    # Block Logic: Updates the device's own sensor data with the computed result.
                    # Functional Utility: Reflects local state changes due to script processing.
                    self.device.set_data(location, result)

                # Block Logic: Releases the per-location lock after data operations are complete.
                # Functional Utility: Allows other threads or devices to access the data location.
                self.device.map_locks[location].release()

            # Block Logic: Global synchronization point for all devices across the simulation.
            # Functional Utility: Ensures all devices have completed their processing for the current
            #                      timepoint before advancing to the next.
            self.device.barrier.wait()
            # Block Logic: The first thread (ID 0) resets the device's state for the next timepoint.
            # Functional Utility: Centralizes state management for multi-threaded devices for consistency.
            if self.id == 0:
                # Inline: Resets the script counter to allow processing of new scripts in the next timepoint.
                self.device.counter = 0
                # Inline: Clears the timepoint_done event, preparing it for the next supervisor signal.
                self.device.timepoint_done.clear()
