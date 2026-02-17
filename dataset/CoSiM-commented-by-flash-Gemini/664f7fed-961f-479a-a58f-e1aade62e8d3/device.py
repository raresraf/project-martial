"""
@664f7fed-961f-479a-a58f-e1aade62e8d3/device.py
@brief Implements a simulated device for a distributed sensor network, with concurrent script execution, fine-grained locking, and barrier synchronization.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` that dispatches scripts to `ScriptThread` instances,
limiting concurrency with a `Semaphore`. Synchronization is handled by a `ReusableBarrier`,
and data access is protected by multiple `Lock` objects: general-purpose locks for
`set_data`, `get_data`, `assign_script`, and a dictionary of location-specific locks.
"""

from threading import Event, Thread, Lock, Semaphore
from ReusableBarrier import ReusableBarrier # Assumed to contain ReusableBarrier class.


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, a shared barrier, and multiple layers of locking
    for data consistency and concurrency control.
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
        self.script_received = Event() # Event to signal that a script has been assigned.
        self.scripts = [] # List to store assigned scripts.

        # Block Logic: Locks for protecting specific device operations or data access.
        self.lock_setter = Lock() # Lock for protecting `set_data` operations.
        self.lock_getter = Lock() # Lock for protecting `get_data` operations.
        self.lock_assign = Lock() # Lock for protecting `assign_script` operations.

        # Block Logic: Shared synchronization primitives.
        self.barrier = None # Shared ReusableBarrier for global time step synchronization.
        self.location_lock = {} # Dictionary to hold `Lock` objects for each sensor data location.

        # Block Logic: Semaphore to control the number of concurrent `ScriptThread` executions.
        self.semaphore = Semaphore(8) # Limits to 8 concurrent script threads.


        self.thread = DeviceThread(self) # The dedicated thread for this device.

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources like the barrier and location-specific locks among devices.
        Only the device with `device_id == 0` initializes these shared resources,
        which are then distributed to all other devices. Also starts the `DeviceThread` for each device.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: The device with `device_id == 0` initializes the shared barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))

            # Block Logic: Iterates through all devices to collect unique sensor data locations
            # and create shared `Lock` objects for each.
            for device in devices[:]: # Iterate over a copy to avoid modification issues if `devices` is modified during loop.
                for loc in device.sensor_data.keys():
                    if loc not in self.location_lock:
                        self.location_lock[loc] = Lock()

            # Block Logic: Distributes the initialized shared barrier and location-specific locks to all devices,
            # and starts each device's main thread.
            for device in devices[:]:
                device.barrier = self.barrier
                device.location_lock = self.location_lock
                
                device.thread.start()


    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Protected by `self.lock_assign`. If no script is provided, it signals `script_received`.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        # Block Logic: Acquires `lock_assign` to protect the process of assigning scripts.
        with self.lock_assign:

            if script is not None:
                self.scripts.append((script, location))
            else:
                # Block Logic: If no script is provided (e.g., end of scripts for timepoint), signals `script_received`.
                self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location, protected by `self.lock_getter`.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        # Block Logic: Acquires `lock_getter` to protect reading from `sensor_data`.
        with self.lock_getter:

            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location, protected by `self.lock_setter`.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        # Block Logic: Acquires `lock_setter` to protect writing to `sensor_data`.
        with self.lock_setter:

            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its graceful completion.
        """
        self.thread.join()


class ScriptThread(Thread):
    """
    @brief A dedicated thread for executing a single script for a specific data location.
    This thread is responsible for gathering data, running the script, and then
    propagating the results to relevant devices, ensuring thread-safe access to data
    through location-specific locks and concurrency control with a semaphore.
    """
    

    def __init__(self, device_thread, script, location, neighbours):
        """
        @brief Initializes a `ScriptThread` instance.
        @param device_thread: The parent `DeviceThread` instance that created this script thread.
        @param script: The script object to execute.
        @param location: The data location that the script operates on.
        @param neighbours: A list of neighboring `Device` instances.
        """
        Thread.__init__(self)
        self.script = script
        self.device_thread = device_thread


        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for `ScriptThread`.
        Block Logic:
        1. Acquires the location-specific lock for the current script's data `location`.
        2. Acquires a permit from the global semaphore (thread pool control).
        3. Collects data from neighboring devices and its own device for the specified `location`.
        4. Executes the assigned `script` if any data was collected.
        5. Propagates the script's `result` to neighboring devices and its own device.
        6. Releases the permit to the global semaphore.
        7. Releases the location-specific lock.
        Invariant: All data access and modification for a given `location` are protected by a shared `Lock`,
        and overall script concurrency is limited by a `Semaphore`.
        """
        # Block Logic: Acquires the location-specific lock to ensure exclusive access to data at this `location`.
        self.device_thread.device.location_lock[self.location].acquire()

        # Block Logic: Acquires a permit from the global semaphore, limiting concurrent script executions.
        self.device_thread.device.semaphore.acquire()

        script_data = []
        
        # Block Logic: Collects data from neighboring devices for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collects data from its own device for the specified location.
        data = self.device_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script if any data was collected and propagates the result.
        if script_data != []:
            
            result = self.script.run(script_data)

            # Block Logic: Updates neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Block Logic: Updates its own device's data with the script's result.
            self.device_thread.device.set_data(self.location, result)

        # Block Logic: Releases a permit back to the global semaphore.
        self.device_thread.device.semaphore.release()
        # Block Logic: Releases the location-specific lock after all data operations for this script are complete.
        self.device_thread.device.location_lock[self.location].release()




class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts concurrently via `ScriptThread` instances (with a limited thread pool),
    and coordinating with other device threads using a shared `ReusableBarrier`.
    Time Complexity: O(T * S_total * (N * D_access + D_script_run)) where T is the number of timepoints,
    S_total is the total number of scripts executed by the device, N is the number of neighbors,
    D_access is data access time, and D_script_run is script execution time.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Waits for the `script_received` event to be set, indicating that scripts are ready to be processed.
        3. Creates and starts `ScriptThread` instances for each assigned script. Concurrency is limited
           by the global `Semaphore` in `Device`.
           Invariant: All scripts for the current timepoint are submitted for concurrent execution.
        4. Waits for all `ScriptThread` instances to complete.
        5. Clears the `script_received` event for the next timepoint.
        6. Synchronizes with all other device threads using a shared `ReusableBarrier`.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break


            # Block Logic: Waits until the device has received all scripts for the current timepoint.
            self.device.script_received.wait()
            script_threads = [] # List to keep track of active `ScriptThread` instances.

            # Block Logic: Creates and starts `ScriptThread` instances for each assigned script.
            # Concurrency is controlled by `self.device.semaphore`.
            for (script, location) in self.device.scripts:
                
                thread = ScriptThread(self, script, location, neighbours)
                script_threads.append(thread)
                thread.start()

            # Block Logic: Waits for all initiated `ScriptThread` instances to complete their execution.
            for thread in script_threads:
                thread.join()

            # Block Logic: Clears the `script_received` event for the next timepoint cycle.
            self.device.script_received.clear()

            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.barrier.wait()
