"""
@file device.py
@brief Implements a simulated distributed device using custom barriers and fine-grained locking for concurrent script execution.

This module defines the `Device` and `DeviceThread` classes, which together
simulate a node in a distributed sensing network. Each `Device` manages its
local sensor data and processes assigned scripts through a dedicated thread.
It utilizes two custom `RBarrier` instances for timepoint synchronization
and script synchronization, along with an array of `Lock` objects for
fine-grained control over access to sensor data locations.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It also holds references to shared
  synchronization objects (two global barriers and an array of locks).
- `DeviceThread`: A dedicated thread per `Device` that executes the simulation
  logic for each timepoint, including neighborhood discovery, waiting on
  synchronization barriers, and processing assigned scripts.
- `RBarrier`: A reusable barrier implementation used for synchronizing
  multiple threads/devices at specific points in the simulation.

Patterns:
- Barrier Synchronization: Employs two distinct barriers (`time_bar` and `script_bar`)
  to coordinate devices at different stages of a simulation timepoint.
- Fine-grained Locking: Uses an array of `Lock` objects (`devloc`) to protect
  access to individual sensor data locations, preventing race conditions
  during concurrent reads/writes.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `DeviceThread` acts as a consumer, processing them.
"""

from threading import Event, Thread, Lock
from barrier import RBarrier


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, assigned scripts, and coordinates its operation
    within the simulated environment using a dedicated thread and shared
    synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing initial sensor readings
                            for various locations.
        @param supervisor: A reference to the central supervisor managing
                           the distributed system.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal that scripts are ready to be processed
        self.scripts = [] # List to store assigned scripts
        self.timepoint_done = Event() # Event for general timepoint completion (unused in this version)

        self.thread = DeviceThread(self) # Dedicated thread for this device's operations
        self.thread.start() # Starts the device's main thread

        self.time_bar = None         # Global barrier for timepoint synchronization
        self.script_bar = None       # Global barrier for script assignment synchronization
        self.devloc = []             # Array of locks for individual sensor data locations

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device communication and synchronization mechanisms.

        If this is the first device (device_id 0), it initializes the global
        `time_bar` and `script_bar` (RBarrier instances) and an array of `Lock` objects
        (`devloc`) for fine-grained data access. These shared synchronization
        primitives are then propagated to all other devices. It also determines
        the maximum sensor location index to size the `devloc` array.

        @param devices: A list of all Device instances in the simulation.
        """

        # Block Logic: Initializes shared synchronization resources if this is the first device (device_id 0).
        if self.device_id == 0:
            # Functional Utility: Initializes two global reusable barriers for synchronization:
            # `time_bar` for overall timepoint synchronization, and `script_bar` for script processing phases.
            self.time_bar = RBarrier(len(devices))
            self.script_bar = RBarrier(len(devices))

            # Block Logic: Propagates the initialized shared barriers to all devices in the simulation.
            for device in devices:
                device.time_bar = self.time_bar
                device.script_bar = self.script_bar

            # Block Logic: Determines the maximum sensor location index across all devices
            # to correctly size the array of per-location locks (`devloc`).
            maxim = 0
            for device in devices:
                loc_list = device.sensor_data.keys()
                loc_list.sort() # Sorting is needed to get the last element (max)
                if loc_list and loc_list[-1] > maxim: # Ensure loc_list is not empty
                    maxim = loc_list[-1]

            # Block Logic: Initializes an array of `Lock` objects, one for each possible sensor location.
            # This allows for fine-grained locking of individual sensor data entries.
            while maxim >= 0:
                self.devloc.append(Lock())
                maxim = maxim - 1

            # Block Logic: Propagates the initialized array of location-specific locks to all devices.
            for device in devices:
                device.devloc = self.devloc


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script object is provided, it is added to the device's list of scripts.
        If no script is provided (None), it signals that scripts for the current
        timepoint are fully received (`script_received` event is set) and then
        all devices wait on the `script_bar` to ensure all scripts are assigned
        before processing begins.

        @param script: The script object to execute, or None to signal script assignment completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """

        # Block Logic: Handles script assignment and synchronization.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: Signals that all scripts for the current timepoint have been assigned.
            self.script_received.set()
            # Block Logic: Synchronizes all devices to ensure that all scripts have been
            # assigned across the entire system before any device proceeds to execute them.
            self.script_bar.wait()


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

    This thread is responsible for discovering neighbors, waiting on various
    synchronization barriers, and processing assigned scripts for each timepoint.
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

        Continuously performs neighborhood discovery, waits on the `script_received`
        event and `script_bar` for script readiness, then waits on `time_bar` for
        timepoint synchronization. It then processes all assigned scripts, acquiring
        and releasing per-location locks, and finally uses `time_bar` for global
        timepoint synchronization.
        """
        while True:

            # Block Logic: Discovers neighboring devices for the current timepoint.
            # Pre-condition: `self.device.supervisor` is available to provide neighborhood information.
            neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Waits until the `script_received` event is set by the supervisor,
            # indicating that script assignment for the current timepoint is complete.
            self.device.script_received.wait()

            # Block Logic: Synchronizes all devices on `script_bar` to ensure all
            # devices have received their scripts before any proceed.
            self.device.script_bar.wait()

            # Block Logic: Synchronizes all devices on `time_bar` before starting
            # script execution for the current timepoint.
            self.device.time_bar.wait()

            # Block Logic: Checks for a shutdown condition (None neighbors indicates termination).
            if neighbours is None:
                break

            # Block Logic: Iterates through all assigned scripts for the current timepoint.
            for (script, location) in self.device.scripts:

                # Block Logic: Acquires the location-specific lock to ensure exclusive access
                # to the sensor data at this `location` during script execution.
                self.device.devloc[location].acquire()

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

                # Block Logic: Releases the location-specific lock, allowing other threads to access it.
                self.device.devloc[location].release()
