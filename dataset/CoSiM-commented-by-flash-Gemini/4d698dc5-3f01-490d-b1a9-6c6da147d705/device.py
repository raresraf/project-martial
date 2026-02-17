"""
@4d698dc5-3f01-490d-b1a9-6c6da147d705/device.py
@brief Implements a simulated device for a distributed sensor network, with dynamic, location-based locking and barrier synchronization.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` for operational logic and uses a `ReusableBarrierSem`
for global time-step synchronization. A unique aspect is the dynamic creation and
sharing of `Lock` objects (`self.locks`) for each data `location` across all devices.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, a shared barrier, and dynamically managed
    location-specific locks.
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
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Initializes a list to hold location-specific locks. Assumes locations are numerical indices.
        # This list will be populated dynamically.
        self.locks = [None] * 100 
        self.devices = None # Will store a reference to all devices in the simulation.
        self.barrier = None # Shared barrier for global time step synchronization.

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared reusable barrier for synchronization among all devices.
        Initializes the global barrier if it hasn't been set yet and distributes it to all devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: Stores a reference to all devices.
        self.devices = devices
        # Block Logic: Initializes the shared barrier if it has not been set yet, and distributes it to all devices.
        # Invariant: A single `ReusableBarrierSem` instance is created and shared across all devices.
        if self.barrier is None:
            self.barrier = ReusableBarrierSem(len(devices))
            for i in self.devices:
                i.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Dynamically creates and shares a `Lock` for the given `location` across all devices
        if one doesn't already exist.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            # Block Logic: Dynamically creates a lock for the given location if it doesn't exist.
            # This lock is then shared among all devices that access this location.
            # Invariant: Each unique `location` will have one shared `Lock` object.
            if self.locks[location] is None:
                self.locks[location] = Lock()
                for i in self.devices:
                    i.locks[location] = self.locks[location]

            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Block Logic: Signals completion of timepoint setup if no script is assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
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
    executing scripts, and coordinating with other device threads using a shared barrier
    and location-specific locks.
    Time Complexity: O(T * S * (N * D_access + D_script_run)) where T is the number of timepoints,
    S is the number of scripts per device, N is the number of neighbors, D_access is data access
    time, and D_script_run is script execution time.
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
        3. Executes assigned scripts: for each script, it acquires the location-specific lock to get data
           from neighbors and itself, runs the script, and then releases the lock after updating data.
           Invariant: Data access and modification for a given location are protected by its corresponding lock.
        4. Clears the `timepoint_done` event for the next timepoint.
        5. Synchronizes with all other device threads using a shared barrier.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()

            # Block Logic: Processes each script assigned to the device for the current timepoint.
            # Invariant: Each script retrieves data from neighbors and itself, executes, and updates data,
            # all while holding the appropriate location-specific lock.
            for (script, location) in self.device.scripts:
                script_data = []

                # Block Logic: Acquires the lock specific to the data location to ensure exclusive access.
                self.device.locks[location].acquire()
                
                # Block Logic: Collects data from neighboring devices for the specified location.
                for device in neighbours:
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
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Updates its own device's data with the script's result.
                    self.device.set_data(location, result)
                # Block Logic: Releases the location-specific lock after all data operations for this script are complete.
                self.device.locks[location].release()

            # Block Logic: Clears the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.barrier.wait()
