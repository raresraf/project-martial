"""
@5b4930bf-6b41-4c98-ba17-21dcd3bf7cac/device.py
@brief Implements a simulated device for a distributed sensor network, with concurrent script execution and barrier synchronization.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` for operational logic and uses a `ReusableBarrierSem`
for global time-step synchronization. Scripts are executed concurrently by `MyThread` instances.
"""

from threading import Event, Thread
from barrier import *

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread and a shared barrier.
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
        self.script_received = Event() # Event to signal that a script has been assigned (though not explicitly used in DeviceThread).
        self.scripts = [] # List to store assigned scripts.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.


        self.thread = DeviceThread(self)
        self.thread.start()
        self.devices = [] # Will store a reference to all devices in the simulation.
        self.barrier = None # Shared barrier for global time step synchronization.
        self.threads = [] # List to keep track of active `MyThread` instances.

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared `ReusableBarrierSem` for synchronization among all devices.
        If not already set, it initializes the global barrier and distributes it to all devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: Initializes the shared barrier if it has not been set yet, and distributes it to all devices.
        # Invariant: A single `ReusableBarrierSem` instance is created and shared across all devices.
        if self.barrier is None:
            barrier = ReusableBarrierSem(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
        # Block Logic: Stores a reference to all devices.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        If a script is provided, it's added to the queue, and `script_received` is set.
        If no script (i.e., `None`) is provided, it signals that the timepoint is done.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Block Logic: Signals completion of the timepoint if no script is assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire any locks, which could lead to race conditions
        if `sensor_data` is modified concurrently by another thread.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire any locks, which could lead to race conditions
        if `sensor_data` is modified concurrently by another thread.
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
    executing scripts concurrently via `MyThread` instances, and coordinating with
    other device threads using a shared `ReusableBarrierSem`.
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
        3. Creates and starts a `MyThread` for each assigned script, allowing concurrent execution.
           Invariant: All scripts for the current timepoint are executed in parallel.
        4. Waits for all `MyThread` instances to complete.
        5. Clears the `timepoint_done` event for the next timepoint.
        6. Synchronizes with all other device threads using a shared `ReusableBarrierSem`.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()

            # Block Logic: Creates and starts `MyThread` instances for each assigned script.
            # `MyThread` is assumed to be defined elsewhere or is a generic Thread.
            for (script, location) in self.device.scripts:
                mythread = MyThread(self.device, location, script, neighbours) # Assuming MyThread constructor
                self.device.threads.append(mythread)

            # Block Logic: Starts all created `MyThread` instances.
            for xthread in self.device.threads:
                xthread.start()
            # Block Logic: Waits for all `MyThread` instances to complete their execution.
            for xthread in self.device.threads:
                xthread.join()

            
            self.device.threads = [] # Clears the list of threads.
            
            # Block Logic: Clears the `timepoint_done` event for the next timepoint cycle.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.barrier.wait()
