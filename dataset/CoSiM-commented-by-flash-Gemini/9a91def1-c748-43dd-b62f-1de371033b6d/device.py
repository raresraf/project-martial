"""
@file device.py
@brief Implements a simulated distributed device with a master thread and shared synchronization mechanisms for concurrent script execution.

This module defines the `Device` and `DeviceThread` classes, which together
simulate a node in a distributed sensing network. Each `Device` manages its
local sensor data, executes assigned scripts through a dedicated thread,
and uses shared locks for data consistency across devices. A global barrier
is employed for synchronizing all devices at the end of each simulation timepoint.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and references to shared synchronization objects (locks and barrier).
- `DeviceThread`: A dedicated thread per `Device` that executes the simulation
  logic for each timepoint, including neighborhood discovery, script processing,
  and device-level synchronization.
- `ReusableBarrierSem`: A semaphore-based barrier for global synchronization
  across all devices at each simulation step.
- `Lock`: Used for fine-grained locking of sensor data at specific locations,
  ensuring data consistency during concurrent access.

Patterns:
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `DeviceThread` acts as a consumer, processing them.
- Barrier Synchronization: Ensures all devices complete a timepoint's processing
  before proceeding to the next.
- Fine-grained Locking: Ensures data consistency when multiple devices (via their
  threads) access shared sensor data locations.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

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
        self.script_received = Event() # Event to signal when scripts are assigned for a timepoint
        self.scripts = [] # List to store assigned scripts
        self.timepoint_done = Event() # Event to signal when timepoint processing is ready to proceed
        self.thread = DeviceThread(self) # Dedicated thread for this device
        self.thread.start() # Starts the device's main thread
        self.locks = [None] * 100 # Array of locks for specific sensor data locations
        self.devices = None # Reference to the list of all devices in the simulation
        self.barrier = None # Global barrier for synchronizing all devices

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device communication and synchronization mechanisms.

        Initializes the global barrier if it hasn't been set up yet, and ensures
        all devices share the same barrier instance. It also stores a reference
        to all devices in the simulation.

        @param devices: A list of all Device instances in the simulation.
        """
        self.devices = devices
        # Block Logic: Initializes a global synchronization barrier for all devices if not already present.
        # Pre-condition: This block ensures that a single barrier instance is shared across all devices.
        if self.barrier is None:
            self.barrier = ReusableBarrierSem(len(devices))
            # Block Logic: Propagates the shared barrier instance to all other devices.
            for i in self.devices:
                i.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the device's script list.
        A shared lock for the specified location is created if it doesn't exist
        and propagated to all devices. If no script is provided (None), it signals
        that the current timepoint's script assignments are complete.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """
        # Block Logic: Handles script assignment and prepares synchronization objects.
        if script is not None:
            # Block Logic: Initializes a location-specific lock if one does not exist.
            # This lock is then shared across all devices to protect access to this location.
            if self.locks[location] is None:
                self.locks[location] = Lock()
                for i in self.devices:
                    i.locks[location] = self.locks[location]

            self.scripts.append((script, location))
            self.script_received.set() # Signals the device thread that scripts have been received
        else:
            self.timepoint_done.set() # Signals the device thread that script assignment for timepoint is done

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

    This thread is responsible for discovering neighbors, executing assigned
    scripts, and synchronizing with other devices at each timepoint.
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

        Continuously performs neighborhood discovery, waits for timepoint
        completion signals, processes all assigned scripts for the timepoint
        with appropriate locking, and then synchronizes with other devices
        globally using a barrier.
        """
        while True:
            # Block Logic: Discovers neighboring devices for the current timepoint.
            # Pre-condition: `self.device.supervisor` is available to provide neighborhood information.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks for a shutdown condition (None neighbors indicates termination).
            if neighbours is None:
                break

            # Block Logic: Waits for the supervisor to signal that script assignments for
            # the current timepoint are complete.
            # Invariant: All scripts for the current timepoint are in `self.device.scripts` after this wait.
            self.device.timepoint_done.wait()

            # Block Logic: Processes all assigned scripts for the current timepoint.
            # Each script is executed on relevant sensor data, with proper locking.
            for (script, location) in self.device.scripts:
                script_data = []

                # Block Logic: Acquires a location-specific lock to ensure exclusive access
                # to the sensor data at this location during script execution.
                self.device.locks[location].acquire()
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
                # Block Logic: Releases the location-specific lock, allowing other devices/threads to access the data.
                self.device.locks[location].release()

            # Block Logic: Clears the timepoint_done event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes all devices across the simulation, ensuring all
            # have completed their timepoint processing before proceeding.
            self.device.barrier.wait()
