"""
@60fc9194-d40f-40d6-9dd3-ed2b631770b2/device.py
@brief Implements a simulated device in a distributed sensor network, with multi-threaded script execution and hierarchical locking.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThreadPool` (acting as the main device thread) which manages
concurrent `DeviceThread` instances for script execution. Synchronization is handled
by a global `ReusableBarrier`, a shared `Lock` for general device-wide protection,
and a `lock_map` for location-specific locks, ensuring fine-grained data consistency.
"""

from threading import Thread, Lock
from barrier import ReusableBarrier # Assumed to contain a ReusableBarrier class.


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a main thread pool, individual script threads, shared barriers, and multiple lock mechanisms.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing the device's local sensor readings.
        @param supervisor: The supervisor object responsible for managing the overall simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to store assigned scripts.
        self.thread = DeviceThreadPool(self) # The main thread for this device, managing script execution.
        
        self.barrier = None # Shared ReusableBarrier for global time step synchronization.
        
        self.inner_barrier = ReusableBarrier(2) # Internal barrier for DeviceThreadPool's synchronization.
        
        self.lock = None # Shared Lock for general device-wide protection of operations.
        
        self.inner_lock = Lock() # Lock for protecting this device's own sensor_data.
        
        # Dictionary to hold Lock objects for each sensor data location, shared across devices.
        self.lock_map = None

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (global barrier, shared lock, and location-specific lock map)
        among all devices. The device with the minimum `device_id` (leader) initializes these resources.
        All devices then start their main thread pool.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: Identifies the leader device (minimum device_id) to initialize shared resources.
        device_ids = [device.device_id for device in devices]
        leader_id = min(device_ids)

        # Block Logic: The leader device initializes the shared barrier, general lock, and location lock map.
        if self.device_id == leader_id:
            barrier = ReusableBarrier(len(devices)) # Global barrier for all devices.
            lock = Lock() # Global lock for device-wide operations.
            lock_map = {} # Map for location-specific locks.
            # Block Logic: Distributes the initialized shared resources to all devices.
            for device in devices:
                device.set_barrier(barrier)
                device.set_lock(lock)
                device.set_lock_map(lock_map)
                device.thread.start() # Start the main device thread for each device.

    def set_barrier(self, barrier):
        """
        @brief Sets the shared `ReusableBarrier` for this device.
        @param barrier: The shared `ReusableBarrier` instance.
        """
        self.barrier = barrier

    def set_lock(self, lock):
        """
        @brief Sets the shared general `Lock` for this device.
        @param lock: The shared `Lock` instance.
        """
        self.lock = lock

    def set_lock_map(self, lock_map):
        """
        @brief Sets the shared `lock_map` for this device.
        @param lock_map: The shared dictionary of location-specific `Lock` objects.
        """
        self.lock_map = lock_map

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        If a new location is encountered, it dynamically creates and adds a new `Lock`
        to the shared `lock_map` for that location, protected by the general device lock.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))

            # Block Logic: Protects access to `lock_map` for adding new location-specific locks.
            with self.lock:
                if location not in self.lock_map:
                    self.lock_map[location] = Lock()
        else:
            # Block Logic: If no script is assigned, signals the `inner_barrier`,
            # indicating completion of script assignment phase.
            self.inner_barrier.wait()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire `self.inner_lock` or `lock_map` locks,
        as it is expected that external mechanisms (e.g., `DeviceThread`) will
        handle the locking for safe data access.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location, protected by `self.inner_lock`.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            # Block Logic: Acquires `self.inner_lock` to protect the device's own `sensor_data` during modification.
            with self.inner_lock:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's main thread pool, waiting for its graceful completion.
        """
        self.thread.join()


class DeviceThreadPool(Thread):
    """
    @brief The main thread of execution for a `Device` instance, managing script execution.
    Despite its name, this class acts as the primary thread for a device, responsible for
    orchestrating the fetching of neighbor data, launching individual `DeviceThread` instances
    for script processing, and global synchronization.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a `DeviceThreadPool` instance.
        @param device: The `Device` instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main loop for the device's operational thread pool.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor, protected by the global device lock.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Synchronizes with another internal barrier (`inner_barrier`) to ensure all scripts for a timepoint are assigned.
        3. Creates and starts a `DeviceThread` for each assigned script, allowing concurrent execution.
           Invariant: All scripts for the current timepoint are executed concurrently.
        4. Waits for all `DeviceThread` instances to complete.
        5. Synchronizes with all other `DeviceThreadPool` instances using the shared global barrier.
           Invariant: All active `DeviceThreadPool` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            # Block Logic: Acquires the global device lock before fetching neighbors to protect shared data.
            with self.device.lock:
                neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break

            # Block Logic: Waits on an internal barrier to ensure script assignments are complete for the timepoint.
            self.device.inner_barrier.wait()

            threads = [] # List to keep track of currently running script execution threads.

            # Block Logic: Iterates through assigned scripts, creating and starting a new `DeviceThread` for each.
            # Invariant: Each script is executed concurrently in its own `DeviceThread`.
            for (script, location) in self.device.scripts:
                thread = DeviceThread(self.device, script, location, neighbours)
                thread.start()
                threads.append(thread)

            # Block Logic: Waits for all `DeviceThread` instances to complete their execution.
            for thread in threads:
                thread.join()

            # Block Logic: Synchronizes with all other `DeviceThreadPool` instances using the shared global barrier,
            # ensuring all devices complete their processing before proceeding to the next timepoint.
            self.device.barrier.wait()


class DeviceThread(Thread):
    """
    @brief A dedicated thread for executing a single script for a specific data location within a device.
    This thread is responsible for gathering data, running the script, and then
    propagating the results to relevant devices, ensuring thread-safe access to data
    through location-specific locks.
    """
    

    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes a `DeviceThread` instance for executing a single script.
        @param device: The parent `Device` instance for which the script is being run.
        @param script: The script object to execute.
        @param location: The data location that the script operates on.
        @param neighbours: A list of neighboring `Device` instances.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for `DeviceThread`.
        Block Logic:
        1. Acquires the location-specific lock for the current script's data `location`.
        2. Collects data from neighboring devices and its own device for the specified `location`.
        3. Executes the assigned `script` if any data was collected.
        4. Propagates the script's `result` to neighboring devices and its own device.
        5. Releases the location-specific lock.
        Invariant: All data access and modification for a given `location` are protected by a shared `Lock` from `lock_map`.
        """
        # Block Logic: Acquires the location-specific lock from the shared `lock_map`
        # to ensure exclusive access to data at this `location` during script execution.
        with self.device.lock_map[self.location]:

            script_data = []
            # Block Logic: Collects data from neighboring devices for the specified location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Collects data from its own device for the specified location.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script if any data was collected and propagates the result.
            if len(script_data) != 0:
                
                result = self.script.run(script_data)

                # Block Logic: Updates neighboring devices with the script's result.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                # Block Logic: Updates its own device's data with the script's result.
                self.device.set_data(self.location, result)
