"""
@7decae1b-20f6-4557-9fae-6c78bcfffb16/device.py
@brief Implements a simulated device for a distributed sensor network, with concurrent script execution and detailed locking mechanisms.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` that dispatches scripts to `InsideDeviceThread` instances
for concurrent processing. Synchronization is handled by a `ReusableBarrierSem` for global
time-step coordination, and a pre-allocated list of `Lock` objects (`lock_for_data`)
provides per-location data protection across devices.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem # Assumed to contain ReusableBarrierSem class.

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, a shared barrier, and a list of location-specific locks.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing the device's local sensor readings.
        @param supervisor: The supervisor object responsible for managing the overall simulation.
        """
        # Block Logic: Initializes a pre-allocated list of Locks for location-specific data protection.
        self.lock_for_data = [None] * 100 
        self.inside_threads = [] # List to keep track of active `InsideDeviceThread` instances.
        self.stored_devices = [] # List to store a reference to all devices in the simulation.
        self.barrier = None # Shared `ReusableBarrierSem` for global time step synchronization.

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal that a script has been assigned.
        self.scripts = [] # List to store assigned scripts.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.thread = DeviceThread(self) # The dedicated thread for this device.
        self.thread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (global barrier and location-specific locks) among all devices.
        Only the device with `device_id == 0` initializes these resources.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: Initializes the `lock_for_data` list with `Lock` objects.
        # This list provides locks for data locations up to index 99.
        for i in range(100):
            self.lock_for_data[i] = Lock()
        
        # Block Logic: Initializes the shared `ReusableBarrierSem` with the total number of devices.
        barrier = ReusableBarrierSem(len(devices))
        
        # Block Logic: Stores a reference to all devices and distributes the initialized shared barrier.
        for device in devices:
            device.barrier = barrier
            self.stored_devices.append(device) # Populates `self.stored_devices` list.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Distributes the `lock_for_data` list to all stored devices, then signals `script_received`.
        If no script is provided, it signals `timepoint_done`.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            
            self.scripts.append((script, location))
            
            # Block Logic: This loop assigns `self.lock_for_data` (which contains locks)
            # from the current device to all `stored_devices`. This seems to be an attempt
            # to share the single list of locks across all devices.
            for device in self.stored_devices:
                self.lock_for_data = device.lock_for_data # Potentially overwrites `self.lock_for_data` for current device.
            
            self.script_received.set() # Signals that a script has been received.

        else:
            # Block Logic: Signals completion of the timepoint if no script is assigned.
            self.timepoint_done.set()


    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `InsideDeviceThread` will acquire the appropriate `lock_for_data` before calling this method.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `InsideDeviceThread` will acquire the appropriate `lock_for_data` before calling this method.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its graceful completion.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts concurrently via `InsideDeviceThread` instances, and coordinating
    with other device threads using a shared `ReusableBarrierSem`.
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
        2. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.
        3. Creates and starts `InsideDeviceThread` instances for each assigned script, allowing concurrent execution.
           Invariant: All scripts for the current timepoint are executed in parallel.
        4. Waits for all `InsideDeviceThread` instances to complete.
        5. Clears the `inside_threads` list for the next timepoint.
        6. Clears the `timepoint_done` event for the next cycle.
        7. Synchronizes with all other device threads using a shared `ReusableBarrierSem`.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()

            # Block Logic: Iterates through assigned scripts, creating and starting an `InsideDeviceThread` for each.
            # Invariant: Each script is executed concurrently in its own `InsideDeviceThread`.
            for (script, location) in self.device.scripts:
                
                inside_thread = InsideDeviceThread(self.device, script, location, neighbours)
                
                self.device.inside_threads.append(inside_thread)
                
                inside_thread.start()

            # Block Logic: Waits for all initiated `InsideDeviceThread` instances to complete their execution.
            for inside_thread in self.device.inside_threads:
                inside_thread.join()

            # Block Logic: Clears the list of `InsideDeviceThread` instances for the next timepoint.
            del self.device.inside_threads[:]
            
            # Block Logic: Clears the `timepoint_done` event for the next timepoint cycle.
            self.device.timepoint_done.clear()
            
            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.barrier.wait()


class InsideDeviceThread(Thread):
    """
    @brief A dedicated thread for executing a single script for a specific data location.
    This thread is responsible for gathering data, running the script, and then
    propagating the results to relevant devices, ensuring thread-safe access to data
    through location-specific `Lock` objects.
    """
    

    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes an `InsideDeviceThread` instance.
        @param device: The parent `Device` instance for which the script is being run.
        @param script: The script object to execute.
        @param location: The data location that the script operates on.
        @param neighbours: A list of neighboring `Device` instances.
        """
        Thread.__init__(self, name="Inside Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours


    def run(self):
        """
        @brief The main execution logic for `InsideDeviceThread`.
        Block Logic:
        1. Acquires the location-specific lock (`lock_for_data[self.location]`) to ensure exclusive access to that data.
        2. Waits for the `script_received` event to be set (although this event is set by `Device.assign_script`,
           waiting here might cause deadlocks if `assign_script` is called once for all scripts and then this thread starts).
        3. Collects data from neighboring devices and its own device for the specified `location`.
        4. Executes the assigned `script` if any data was collected.
        5. Propagates the script's `result` to neighboring devices and its own device.
        6. Releases the location-specific lock.
        Invariant: All data access and modification for a given `location` are protected by a shared `Lock`.
        """
        # Block Logic: Acquires the location-specific lock to ensure exclusive access to data at this `location`.
        self.device.lock_for_data[self.location].acquire()

        # Block Logic: Waits for `script_received` event. This seems to be a coordination point,
        # ensuring scripts are fully assigned before execution, but its interaction with `acquire()`
        # might imply a specific sequence or potential for deadlock if not managed carefully.
        self.device.script_received.wait()

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
        if script_data != []:
            
            result = self.script.run(script_data)

            # Block Logic: Updates neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Block Logic: Updates its own device's data with the script's result.
            self.device.set_data(self.location, result)

        # Block Logic: Releases the location-specific lock after all data operations for this script are complete.
        self.device.lock_for_data[self.location].release()
