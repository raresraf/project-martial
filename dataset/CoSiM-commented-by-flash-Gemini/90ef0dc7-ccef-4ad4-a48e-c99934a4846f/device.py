


"""
@90ef0dc7-ccef-4ad4-a48e-c99934a4846f/device.py
@brief Implements a multi-threaded device simulation with script queuing and conditional barrier synchronization.

This module defines a `Device` that manages its own sensor data and orchestrates parallel script execution
through a main `DeviceThread` and dynamically spawned `WorkerThread`s. Scripts are distributed
via a `Queue`, and synchronization is achieved using a `ReusableBarrierCond` for global timepoint
coordination and per-location locks for data integrity during concurrent updates.
"""

from threading import Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue


class Device(object):
    """
    @brief Represents a simulated device managing sensor data, script queuing, and multi-threaded execution.

    Each Device instance is responsible for holding its unique ID, sensor readings,
    and a reference to the supervisor. It maintains a `Queue` for incoming scripts
    and orchestrates a `DeviceThread` which, in turn, manages multiple `WorkerThread`s
    for concurrent script processing. Synchronization mechanisms like `location_locks`,
    a global lock, and a `ReusableBarrierCond` are managed to ensure data consistency
    across the distributed simulation.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance, setting up its state and multi-threading components.

        Initializes device-specific attributes such as ID, sensor data, and supervisor reference.
        It sets up a `Queue` for scripts, defines the number of worker threads, and
        prepares synchronization primitives including:
        - `location_locks`: Dynamically created locks for specific data locations.
        - `lock`: A global lock for managing access to shared device resources.
        - `barrier`: A global `ReusableBarrierCond` for inter-device synchronization.
        - `thread`: The main `DeviceThread` responsible for orchestrating worker threads.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
        @param supervisor: A reference to the supervisor object managing the device network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.queue = Queue()
        self.num_threads = 8

        self.location_locks = None
        self.lock = None
        self.barrier = None

        self.thread = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures global synchronization resources and starts the device's main thread.

        If this is the device with `device_id == 0` (acting as a coordinator):
        - It initializes the shared `location_locks` dictionary, a global `lock`,
          and a `ReusableBarrierCond` to synchronize all devices in the simulation.
        - These shared resources are then propagated to all other devices.
        Finally, it creates and starts its own `DeviceThread` which will manage the
        worker threads for this specific device.

        @param devices: A list of all Device instances participating in the simulation.
        """
        
        if self.device_id == 0:
            self.location_locks = {}
            self.lock = Lock()
            self.barrier = ReusableBarrierCond(len(devices))


            for device in devices:
                if device.device_id != 0:
                    device.location_locks = self.location_locks
                    device.lock = self.lock
                    device.barrier = self.barrier
        self.thread = DeviceThread(self)
        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the processing queue and manages location-specific locks.

        If a `script` is provided (not None), it's added to the internal `queue` along
        with its `location`. A global lock (`self.lock`) is used to safely check and
        create a new `Lock` for the `location` in `self.location_locks` if one doesn't
        already exist, ensuring concurrent access control for that specific data point.
        If `script` is None, it signals the end of scripts for the current timepoint
        by adding `num_threads` (None, None) tuples to the queue, which act as
        termination signals for the worker threads.

        @param script: The script object to be executed, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) the script operates on.
        """
        
        if script is not None:
            with self.lock:
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()
            self.queue.put((script, location))
        else:
            for _ in range(self.num_threads):
                self.queue.put((None, None))

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The key identifying the sensor data to retrieve.
        @return: The sensor data at the specified location, or None if not found.
        """
        
        return self.sensor_data[
            location] if location in self.sensor_data else None

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
        @brief Shuts down the device's main processing thread.

        This method waits for the device's main `DeviceThread` to complete
        its execution, ensuring a clean and orderly shutdown.
        """
        
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Orchestrates the execution of worker threads for a Device at each timepoint.

    This thread acts as the main control unit for a `Device`, responsible for
    fetching neighborhood information and spawning a pool of `WorkerThread`s
    for each simulation timepoint. It ensures all worker threads complete their
    tasks and then participates in a global barrier synchronization before
    proceeding to the next timepoint.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        Associates the thread with its parent `Device` instance.

        @param device: The parent Device instance this thread belongs to.
        """
        
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        This loop continuously performs the following steps for each simulation timepoint:
        1.  Fetches the current set of neighboring devices from the supervisor.
        2.  Terminates if no neighbors are found, signifying the end of the simulation for this device.
        3.  Spawns a pool of `WorkerThread`s, each tasked with processing scripts from the queue.
        4.  Waits for all `WorkerThread`s to complete their execution for the current timepoint.
        5.  Participates in a global `ReusableBarrierCond` synchronization, ensuring all devices
            are synchronized before advancing to the next timepoint.
        """
        while True:
            # Block Logic: Fetches the current set of active neighbors for data exchange.
            # Functional Utility: Dynamically updates the device's awareness of its network topology.
            neighbours = self.device.supervisor.get_neighbours()
            # Invariant: If no neighbors are returned, the simulation for this device is complete.
            if neighbours is None:
                break

            # Block Logic: Creates and starts a pool of WorkerThread instances for concurrent script processing.
            # Architectural Intent: Leverages multi-threading to parallelize the execution of scripts
            #                      for the current timepoint.
            worker_threads = [WorkerThread(self.device, neighbours) for _ in
                              range(self.device.num_threads)]
            for thread in worker_threads:
                thread.start()
            # Block Logic: Waits for all spawned WorkerThread instances to complete their assigned tasks.
            # Functional Utility: Ensures all scripts for the current timepoint are processed before global synchronization.
            for thread in worker_threads:
                thread.join()

            # Block Logic: Global synchronization point for all devices across the simulation.
            # Functional Utility: Ensures all devices have completed their processing for the current
            #                      timepoint before advancing to the next.
            self.device.barrier.wait()


class WorkerThread(Thread):
    """
    @brief Represents a worker thread that executes individual scripts for a Device.

    These threads are spawned by the `DeviceThread` for each timepoint. Their responsibility
    is to fetch scripts from the device's queue, gather relevant data from neighbors and
    the device itself, execute the script, and then disseminate the results, while
    adhering to location-specific locking for data integrity.
    """
    

    def __init__(self, device, neighbours):
        """
        @brief Initializes a new WorkerThread instance.

        Associates the thread with its parent `Device` instance and a snapshot
        of the current `neighbours` list, which is stable for the duration
        of this worker thread's execution.

        @param device: The parent Device instance this worker thread belongs to.
        @param neighbours: A list of neighboring Device instances relevant for this timepoint.
        """
        
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours

    def run_script(self, script, location):
        """
        @brief Executes a given script, gathering data and disseminating results.

        This helper method performs the core logic of a worker:
        1.  Collects relevant sensor data from all `neighbours` and the worker's
            `device` itself for the specified `location`.
        2.  If data is available, it executes the provided `script` with the
            collected data.
        3.  Disseminates the `result` of the script execution by updating the
            sensor data of all `neighbours` and the worker's own `device` at
            the specified `location`.

        @param script: The script object to be executed.
        @param location: The data location (e.g., sensor ID) the script operates on.
        """
        
        script_data = []
        
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            
            result = script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)

    def run(self):
        """
        @brief The main execution loop for the WorkerThread.

        This loop continuously:
        1.  Retrieves a `(script, location)` pair from the device's shared `queue`.
            If `script` is None, it acts as a termination signal for this worker thread.
        2.  Acquires the location-specific lock (`self.device.location_locks[location]`)
            to ensure exclusive access to data at that location during processing.
        3.  Executes the script by calling `self.run_script(script, location)`.
        4.  Releases the location-specific lock.
        5.  Puts the `(script, location)` back into the queue. This design implies that
            scripts are continuously recycled and processed across multiple timepoints.
        """
        while True:
            # Block Logic: Retrieves a script and its associated location from the device's queue.
            # Functional Utility: Distributes processing tasks (scripts) among available worker threads.
            script, location = self.device.queue.get()
            # Invariant: If a None script is received, it signifies that there are no more scripts to process
            #           for the current timepoint or simulation, and the worker thread terminates.
            if script is None:
                return
            # Block Logic: Acquires a lock specific to the data location before processing the script.
            # Functional Utility: Ensures atomicity and prevents race conditions for data access/modification
            #                      at a given sensor location by multiple concurrent workers.
            with self.device.location_locks[location]:
                self.run_script(script, location)
            # Block Logic: Re-enqueues the processed script for potential future execution.
            # Architectural Intent: Supports a continuous processing model where scripts may be
            #                      re-evaluated across multiple timepoints or cycles.
            self.device.queue.put((script, location))
