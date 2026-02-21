"""
@file device.py
@brief Implements a simulated distributed device using an external reusable barrier and a single thread for script execution per device.

This module defines the `Device` and `DeviceThread` classes.
The `Device` class represents a node in a simulated sensing network, managing
local sensor data and coordinating script execution through a dedicated
`DeviceThread`. It leverages an external `ReusableBarrier` for global
timepoint synchronization.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It registers itself with a shared
  `ReusableBarrier`.
- `DeviceThread`: A dedicated thread per `Device` that executes the simulation
  logic for each timepoint, including neighborhood discovery, waiting on
  synchronization barriers, and processing assigned scripts sequentially.
- `ReusableBarrier`: An external reusable barrier used for global
  synchronization across all devices at each simulation step.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `DeviceThread` acts as a consumer, processing them sequentially.
"""

from threading import Event, Thread
import ReusableBarrier

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, processes assigned scripts sequentially,
    and participates in global synchronization with other devices.
    """

    # Functional Utility: A class-level instance of ReusableBarrier, shared by all Device instances.
    reusable_barrier = ReusableBarrier.ReusableBarrier()

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Registers itself with the global `reusable_barrier`.

        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing initial sensor readings
                            for various locations.
        @param supervisor: A reference to the central supervisor managing
                           the distributed system.
        """

        # Functional Utility: Registers this device's thread with the global reusable barrier.
        Device.reusable_barrier.add_thread()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal that scripts are ready to be processed
        self.scripts = [] # List to store assigned scripts


        self.timepoint_done = Event() # Event to signal that timepoint script assignment is complete (unused in this version)
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations
        self.thread.start() # Starts the device's main thread

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Placeholder method for device setup.

        In this implementation, device setup specific to `Device` instances is not
        explicitly defined here. The global barrier registration happens in `__init__`.
        """

        pass

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script object is provided, it is added to the device's list of scripts.
        If `script` is None, it signals that script assignments for the current
        timepoint are complete by setting the `script_received` event.

        @param script: The script object to execute, or None to signal script assignment completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """
        # Block Logic: Handles script assignment and signals script availability.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set() # Signals the DeviceThread that scripts are ready.


    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The identifier for the data location.
        @return The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The identifier for the data location.
        @param data: The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its main thread to complete.

        Functional Utility: Ensures proper termination and cleanup of resources
        associated with the device's dedicated execution thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device, managing its simulation logic.

    This thread is responsible for discovering neighbors, synchronizing with
    other devices using the global barrier, waiting for script assignments,
    and processing assigned scripts sequentially.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: A reference to the parent Device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Continuously performs neighborhood discovery, synchronizes with other
        devices using the global barrier, waits for script assignments to complete,
        and processes all assigned scripts sequentially.
        """
        while True:
            # Block Logic: Discovers neighboring devices for the current timepoint.
            # Pre-condition: `self.device.supervisor` is available to provide neighborhood information.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks for a shutdown condition (None neighbors indicates termination).
            if neighbours is None:
                break

            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their neighborhood discovery before proceeding.
            Device.reusable_barrier.wait();
            # Block Logic: Waits for the `script_received` event to be set, indicating that
            # all scripts for the current timepoint have been assigned.
            self.device.script_received.wait()
            # Block Logic: Clears the `script_received` event, resetting it for the next timepoint.
            self.device.script_received.clear();

            # Block Logic: Processes all assigned scripts for the current timepoint sequentially.
            # Each script is executed on relevant sensor data.
            for (script, location) in self.device.scripts:
                script_data = []
                # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Block Logic: Includes the device's own sensor data in the script input.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if there is any data to process.
                if script_data != []:
                    # Functional Utility: Executes the assigned script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Propagates the script's result back to neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)

                    # Block Logic: Updates the device's own sensor data with the script's result.
                    self.device.set_data(location, result)