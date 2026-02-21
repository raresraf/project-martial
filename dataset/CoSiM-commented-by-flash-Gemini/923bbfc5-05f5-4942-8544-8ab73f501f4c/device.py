


"""
@923bbfc5-05f5-4942-8544-8ab73f501f4c/device.py
@brief Implements a device simulation with a master-worker threading model for script execution.

This module defines a `Device` class that orchestrates script processing using a dedicated
master thread and a pool of worker threads. It leverages `ReusableBarrierSem` for global
synchronization across devices and `RLock`s for fine-grained, location-specific data access control,
ensuring data consistency in a concurrent and distributed simulation environment.
"""

from threading import Event, Thread, RLock, Semaphore
from barrier import ReusableBarrierSem

class Device(object):
    """
    @brief Represents a simulated device with a master-worker threading architecture for script processing.

    Each Device instance manages its sensor data, interacts with a supervisor for network topology,
    and coordinates script execution. A central 'master' thread ({@code master_func}) spawns
    'worker' threads ({@code worker_func}) to concurrently process scripts. Synchronization is
    achieved through global barriers ({@code step_barrier}) and fine-grained recursive locks
    (RLocks) for data consistency across shared sensor locations.
    """
    

    def __init__(self, device_id, sensor_data, supervisor, max_workers=8):
        """
        @brief Initializes a new Device instance.

        Sets up device-specific attributes such as ID, sensor data, and supervisor reference.
        It also initializes internal state for script management, the main 'master' thread,
        a semaphore to control active worker threads, a reference to the 'root' device
        (for shared resources), a global step barrier, and a dictionary for data locks.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
        @param supervisor: A reference to the supervisor object managing the device network.
        @param max_workers: The maximum number of worker threads that can be active concurrently.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.scripts = []
        self.master = Thread(target=self.master_func)


        self.master.start()
        self.active_workers = Semaphore(max_workers)

        
        self.root_device = None

        
        self.step_barrier = None
        self.data_locks = {}

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures global synchronization resources and identifies the root device.

        This method first identifies the 'root' device (the device with ID 0) among
        all participating `devices` and stores a reference to it.
        If this device *is* the root device, it takes responsibility for:
        - Initializing a global `ReusableBarrierSem` ({@code step_barrier}) to
          synchronize all devices at each simulation step.
        - Initializing `RLock`s for each unique sensor data location across all devices
          to ensure thread-safe access to shared data.

        @param devices: A list of all Device instances participating in the simulation.
        """
        
        for dev in devices:
            if dev.device_id == 0:
                self.root_device = dev

        if self.device_id == 0:
            
            
            
            self.step_barrier = ReusableBarrierSem(len(devices))

            
            for device in devices:
                for (location, _) in device.sensor_data.iteritems():


                    if location not in self.data_locks:
                        self.data_locks[location] = RLock()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be processed or signals script reception completion.

        If a `script` is provided, it is appended to the device's internal `scripts` list.
        If `script` is None, it signals that all scripts for the current timepoint
        have been received by setting the `scripts_received` event.

        @param script: The script object to be executed, or None to signal completion of script assignments.
        @param location: The data location (e.g., sensor ID) the script operates on.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location, ensuring thread safety.

        Access to the sensor data is protected by an {@code RLock} associated with the
        data location, managed by the root device. This prevents race conditions
        during concurrent read operations.

        @param location: The key identifying the sensor data to retrieve.
        @return: The sensor data at the specified location, or None if not found.
        """
        
        with self.root_device.data_locks[location]:
            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location, ensuring thread safety.

        Modification of the sensor data is protected by an {@code RLock} associated with the
        data location, managed by the root device. This prevents race conditions
        during concurrent write operations.

        @param location: The key identifying the sensor data to update.
        @param data: The new data value to set for the specified location.
        """
        
        with self.root_device.data_locks[location]:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's master processing thread.

        This method waits for the device's main 'master' thread to complete
        its execution, ensuring a clean and orderly shutdown.
        """
        
        self.master.join()

    def master_func(self):
        """
        @brief The main execution loop for the master thread of the device.

        This function orchestrates the simulation timepoints by:
        1.  Continuously fetching updated neighborhood information.
        2.  Terminating if no neighbors are found, signaling the end of the simulation.
        3.  Waiting for all scripts for the current timepoint to be assigned.
        4.  Spawning a pool of worker threads to process each script concurrently,
            respecting a maximum number of active workers using a semaphore.
        5.  Waiting for all worker threads to complete their tasks for the current batch of scripts.
        6.  Clearing the script reception event for the next timepoint.
        7.  Participating in a global synchronization barrier (`step_barrier`) to align
            with other devices before proceeding to the next simulation step.
        """
        while True:
            # Block Logic: Fetches the current set of active neighbors from the supervisor.
            # Functional Utility: Dynamically updates the device's awareness of its network topology.
            neighbours = self.supervisor.get_neighbours()
            # Invariant: If no neighbors are returned, the simulation for this device is complete.
            if neighbours is None:
                break

            # Block Logic: Waits for all scripts for the current timepoint to be assigned to this device.
            # Functional Utility: Ensures that the master thread only proceeds to spawn workers once
            #                      all processing instructions for the current step are available.
            self.scripts_received.wait()

            workers = []
            # Block Logic: Iterates through each assigned script and spawns a worker thread to process it.
            # Architectural Intent: Distributes script execution across multiple threads for parallelism.
            for (script, location) in self.scripts:
                # Block Logic: Acquires a permit from the semaphore, limiting the number of concurrent worker threads.
                # Functional Utility: Manages the worker thread pool size to prevent resource exhaustion.
                self.active_workers.acquire()

                # Inline: Creates a new worker thread to execute the specific script.
                worker = Thread(target=self.worker_func, \
                    args=(script, location, neighbours))
                workers.append(worker)
                worker.start()

            # Block Logic: Waits for all worker threads spawned in the current timepoint to complete.
            # Functional Utility: Ensures all scripts for the current timepoint are fully processed
            #                      before signaling completion and moving to the next step.
            for worker in workers:
                worker.join()

            # Block Logic: Resets the script reception event, preparing for the next timepoint's script assignments.
            # Functional Utility: Clears the signal, allowing the event to be set again when new scripts arrive.
            self.scripts_received.clear()
            
            # Block Logic: Global synchronization point for all devices across the simulation.
            # Functional Utility: Ensures all devices have completed their processing for the current
            #                      timepoint before advancing to the next, maintaining simulation consistency.
            self.root_device.step_barrier.wait()


    def worker_func(self, script, location, neighbours):
        """
        @brief Executes a single script within a worker thread context.

        This function represents the task performed by an individual worker thread.
        It encapsulates the process of gathering data, executing a script,
        and disseminating its results, ensuring thread-safe access to shared data
        at specific locations.

        @param script: The script object to be executed.
        @param location: The data location (e.g., sensor ID) the script operates on.
        @param neighbours: A list of neighboring Device instances relevant for this script's execution.
        """
        # Block Logic: Acquires a recursive lock specific to the data location before processing.
        # Functional Utility: Prevents race conditions and ensures atomic updates for data
        #                      at a given sensor location by multiple concurrent workers.
        with self.root_device.data_locks[location]:
            # Block Logic: Gathers relevant sensor data from all specified neighbors for the current script's location.
            # Functional Utility: Collects necessary input for the script based on the current network state.
            script_data = []
            for dev in neighbours:
                data = dev.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: Includes the device's own sensor data for the current script's location.
            # Functional Utility: Ensures the script considers the device's local state.
            data = self.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Block Logic: Executes the assigned script with the aggregated data.
                # Architectural Intent: Decouples computational logic from data management,
                #                      allowing dynamic script execution based on current data.
                result = script.run(script_data)

                # Block Logic: Disseminates the computed result to neighboring devices.
                # Functional Utility: Propagates state changes across the network as a result of script execution.
                for dev in neighbours:
                    dev.set_data(location, result)

                # Block Logic: Updates the device's own sensor data with the computed result.
                # Functional Utility: Reflects local state changes due to script processing.
                self.set_data(location, result)

        # Block Logic: Releases a permit back to the active_workers semaphore.
        # Functional Utility: Signals that this worker thread has completed its task,
        #                      allowing another script to be processed by a new worker.
        self.active_workers.release()
