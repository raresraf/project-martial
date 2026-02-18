"""
@7e9b7078-547d-43a7-ba11-36c82b1e68b7/device.py
@brief Implements a simulated device for a distributed sensor network, with a fixed thread pool for concurrent script execution and hierarchical barrier synchronization.
This module defines a `Device` that processes sensor data and executes scripts.
It features a pool of `DeviceThread` instances for operational logic, using a global
`ReusableBarrierCond` for inter-device synchronization and an internal `ReusableBarrierCond`
for intra-device thread pool synchronization. Data access is protected by a dictionary
of `Lock` objects (`locks`) on a per-location basis.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond # Assumed to contain ReusableBarrierCond class.

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a pool of dedicated threads, a shared global barrier, and location-specific locks.
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
        self.script_received = Event() # Event to signal that a script has been assigned (not explicitly used).
        self.scripts = [] # List to store assigned scripts.
        self.bariera = None # Shared global ReusableBarrierCond for inter-device synchronization.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.threads = [] # List to hold `DeviceThread` instances (the thread pool).
        self.nr_threads = 8 # Number of threads in the internal thread pool.
        self.locks = {} # Dictionary to hold `Lock` objects for each sensor data location.
        # Internal barrier for synchronizing threads within this device's pool.
        self.bariera_interioara = ReusableBarrierCond(self.nr_threads)

        # Block Logic: Creates and starts a fixed pool of `DeviceThread` instances.
        for index in range(0, self.nr_threads): # Changed xrange to range for Python 3 compatibility.
            thread = DeviceThread(self, index)
            self.threads.append(thread)

        for thread in self.threads:
            thread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (global barrier and location-specific locks) among all devices.
        Only the device with `device_id == 0` is responsible for initializing these resources,
        which are then distributed to all other devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        if self.device_id == 0:
            
            # Block Logic: Initializes the shared global `ReusableBarrierCond` with the total number of devices.
            self.bariera = ReusableBarrierCond(len(devices))
            for dev in devices:
                dev.bariera = self.bariera
            
            # Block Logic: Determines the maximum location index across all devices to size the lock map.
            max_location = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > max_location:
                        max_location = location
            
            # Block Logic: Initializes `Lock` objects for each potential location and distributes them to all devices.
            for location in range(0, max_location + 1): # Changed xrange to range for Python 3 compatibility.
                self.locks[location] = Lock()
            for device in devices:
                device.locks = self.locks


    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        If no script is provided, it signals `timepoint_done`, indicating the completion of
        script assignment for the current timepoint.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: Signals completion of the timepoint if no script is assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `DeviceThread` (or `run` method) will acquire the appropriate `locks[location]` before calling this method.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `DeviceThread` (or `run` method) will acquire the appropriate `locks[location]` before calling this method.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down all threads in the device's internal thread pool, waiting for their graceful completion.
        """
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    @brief A worker thread within a device's internal thread pool, responsible for executing assigned scripts.
    Each `DeviceThread` contributes to processing a subset of the device's scripts for a timepoint,
    coordinating with other threads in the pool via `bariera_interioara` and with other devices
    via `bariera`.
    """
    

    def __init__(self, device, index):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The parent `Device` instance that this thread belongs to.
        @param index: The unique index of this thread within the device's thread pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.index = index # Unique index of this thread in the pool.

    def run(self):
        """
        @brief The main execution logic for `DeviceThread`.
        Block Logic:
        1. Continuously synchronizes with other threads in the device's pool (`bariera_interioara`).
        2. If this is the first thread (index 0), it fetches neighbor information from the supervisor.
           Invariant: `self.device.neighbours` is updated once per timepoint by thread 0.
        3. Synchronizes again with the internal barrier. If `neighbours` is `None`, the loop breaks.
        4. If this is the first thread, it waits for `timepoint_done` and then synchronizes globally via `bariera`.
           Invariant: `timepoint_done` and `bariera` are managed by thread 0 for global coordination.
        5. Synchronizes again with the internal barrier.
        6. Processes a subset of scripts assigned to the parent `Device` based on its `index` (modulo distribution).
           For each script, it acquires the location-specific lock, collects data, runs the script,
           updates data on neighbors and itself, and then releases the lock.
           Invariant: Script execution for each location is protected by its respective lock.
        7. If this is the first thread, it clears `timepoint_done` for the next cycle.
        """
        while True:
            # Block Logic: Internal barrier synchronization for all threads in the pool.
            self.device.bariera_interioara.wait()
            # Block Logic: Only the first thread (index 0) fetches neighbor information from the supervisor.
            if self.index == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Internal barrier synchronization before checking for simulation termination.
            self.device.bariera_interioara.wait()
            if self.device.neighbours is None:
                break

            # Block Logic: Only the first thread handles global timepoint synchronization.
            if self.index == 0:
                self.device.timepoint_done.wait() # Waits for the device to be ready for script processing.
                self.device.bariera.wait() # Global barrier synchronization across all devices.
            
            # Block Logic: Internal barrier synchronization before script execution.
            self.device.bariera_interioara.wait()


            # Block Logic: Iterates through assigned scripts and processes only those assigned to this thread (modulo distribution).
            # Invariant: Each thread processes its share of scripts concurrently with other threads in the pool.
            for index in range(0, len(self.device.scripts)): # Changed xrange to range for Python 3 compatibility.
                
                if self.index == index % self.device.nr_threads:
                    (script, location) = self.device.scripts[index]
                    # Block Logic: Acquires the location-specific lock before accessing or modifying data.
                    self.device.locks[location].acquire()
                    script_data = []
                    
                    # Block Logic: Collects data from neighboring devices for the specified location.
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    # Block Logic: Collects data from its own device for the specified location.
                    data = self.device.get_data(location)

                    if data is not None:
                        script_data.append(data)

                    # Block Logic: Executes the script if any data was collected and propagates the result.
                    if script_data != []:
                        
                        result = script.run(script_data)
                        
                        # Block Logic: Updates neighboring devices with the script's result.
                        for device in self.device.neighbours:
                            device.set_data(location, result)
                        
                        # Block Logic: Updates its own device's data with the script's result.
                        self.device.set_data(location, result)
                    # Block Logic: Releases the location-specific lock after data operations are complete.
                    self.device.locks[location].release()
            # Block Logic: Only the first thread clears `timepoint_done` for the next cycle.
            if self.index == 0:
                self.device.timepoint_done.clear()
