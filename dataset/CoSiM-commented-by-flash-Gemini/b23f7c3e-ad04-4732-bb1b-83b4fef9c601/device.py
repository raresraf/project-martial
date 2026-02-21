"""
@file device.py
@brief Implements a simulated distributed device using a reusable barrier and multi-threaded script execution with dynamic fine-grained locking.

This module defines the `Device`, `MyThread`, and `DeviceThread` classes,
which together simulate a node in a distributed sensing network. Each `Device`
manages its local sensor data and executes assigned scripts. A `DeviceThread`
per device coordinates the overall timepoint simulation, while `MyThread`s
are spawned to concurrently execute individual scripts. A shared `ReusableBarrier`
ensures global timepoint synchronization, and a dynamically created list of
`Lock` objects provides fine-grained control over access to sensor data locations.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds references to the global barrier
  and the shared list of data locks.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the
  execution for each timepoint, including neighborhood discovery, spawning
  `MyThread`s, and global timepoint synchronization.
- `MyThread`: Worker threads spawned by `DeviceThread` to execute
  individual scripts. Each script thread acquires a lock for its target
  data location, gathers data from neighbors and itself, runs the script,
  updates data, and releases the lock.
- `ReusableBarrier`: A shared barrier for global synchronization
  across all devices at each simulation step.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `MyThread` instances (spawned by `DeviceThread`) act as consumers,
  processing them.
- Fine-grained Locking: A shared list of `Lock` objects (`locations_lock`) ensures
  exclusive access to sensor data at specific locations during script execution
  across concurrent worker threads.
- Dynamic Lock Creation: Locks for data locations are created dynamically
  based on the maximum location index found in sensor data.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrier


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, processes assigned scripts concurrently,
    and participates in global synchronization with other devices.
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
        self.timepoint_done = Event() # Event to signal that timepoint script assignment is complete

        self.thread = DeviceThread(self) # Dedicated thread for this device's operations

        # Functional Utility: The barrier will be initialized in `setup_devices`.
        self.barrier = ReusableBarrier(0) # Placeholder; actual size determined in setup_devices

        # Functional Utility: List of Locks for fine-grained synchronization on data locations.
        # This list is dynamically sized and populated by the master device.
        self.locations_lock = []

        # Functional Utility: A list to hold `MyThread` instances for concurrent script execution.
        self.thread_list = []

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device synchronization mechanisms.

        If this device is the designated master (device_id 0), it initializes
        the shared `ReusableBarrier` and dynamically creates a shared list of `Lock`
        objects (`locations_lock`) based on the maximum sensor location index across
        all devices. These synchronization primitives are then propagated to all `Device`
        instances. Finally, the device's dedicated thread is started.

        @param devices: A list of all Device instances in the simulation.
        """

        # Block Logic: Initializes the global barrier and the list of locks if this is device 0.
        # This ensures a single set of shared resources is created and distributed.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices)) # Create a single global barrier

            # Block Logic: Determines the maximum sensor location index across all devices
            # to correctly size and initialize the array of per-location locks (`locations_lock`).
            locations = []
            for device in devices:
                if device is not None:
                    # Check if sensor_data is not empty before attempting max()
                    if device.sensor_data:
                        locations.append(max(device.sensor_data.keys()))
            no_locations = 0
            if locations:
                no_locations = max(locations) + 1 # Max location index + 1 for size

            # Block Logic: Initializes `Lock` objects for each potential sensor data location.
            for i in xrange(no_locations):
                self.locations_lock.append(Lock())

            # Block Logic: Propagates the initialized global barrier and list of locks to all devices.
            for device in devices:
                if device is not None:
                    device.barrier = barrier

                    # Block Logic: Shares the dynamically created list of locks with all devices.
                    for i in xrange(no_locations):
                        device.locations_lock.append(self.locations_lock[i])

                    # Functional Utility: Starts the dedicated thread for this device after setup.
                    device.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script object is provided, it is added to the device's list of scripts,
        and the `script_received` event is set. If `script` is None, it signals that
        script assignments for the current timepoint are complete by setting the
        `timepoint_done` event.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a script has been received
        else:
            self.timepoint_done.set() # Signal that all scripts for this timepoint are assigned

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


class MyThread(Thread):
    """
    @brief A worker thread responsible for executing a single script on sensor data.

    Each `MyThread` processes one assigned script, acquiring the necessary
    location-specific lock, gathering data from the local device and its neighbors,
    executing the script, and propagating the results back to the relevant devices.
    """

    def __init__(self, device, neighbours, script, location):
        """
        @brief Initializes a new MyThread instance.

        @param device: A reference to the parent `Device` instance.
        @param neighbours: A list of neighboring `Device` instances.
        @param script: The script object to execute.
        @param location: The data location (e.g., sensor ID) for the script.
        """
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location

    def run(self):
        """
        @brief The main execution logic for the `MyThread`.

        Acquires the location-specific lock, gathers data from neighbors and
        the local device, executes the script, updates sensor data on affected
        devices, and then releases the lock.
        """
        # Block Logic: Acquires the location-specific lock to ensure exclusive access
        # to the sensor data at `self.location` during script execution.
        self.device.locations_lock[self.location].acquire()

        script_data = []
        # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        # Block Logic: Includes the current device's own sensor data in the script input.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script only if there is any data to process.
        if script_data:
            # Functional Utility: Executes the assigned script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Propagates the script's result back to neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)

            # Block Logic: Updates the current device's own sensor data with the script's result.
            self.device.set_data(self.location, result)

        # Block Logic: Releases the location-specific lock, allowing other threads to access it.
        self.device.locations_lock[self.location].release()


class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device, managing its simulation logic.

    This thread is responsible for discovering neighbors, waiting for script
    assignments, spawning `MyThread`s for concurrent script execution,
    and synchronizing globally at timepoint boundaries.
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

        Continuously performs neighborhood discovery, waits for timepoint script
        assignments to complete, spawns `MyThread`s to process scripts concurrently,
        waits for all script threads to finish, and then synchronizes globally
        using the shared barrier before starting the next timepoint.
        """

        # Block Logic: The device's main thread starts its execution loop.
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


            # Functional Utility: List to hold `MyThread` instances for concurrent script execution.
            for (script, location) in self.device.scripts:
                self.device.thread_list.append(MyThread(self.device, neighbours, script, location))

            # Block Logic: Starts all `MyThread` instances.
            for thread in self.device.thread_list:
                thread.start()

            # Block Logic: Waits for all `MyThread` instances to complete their execution.
            for thread in self.device.thread_list:
                thread.join()

            # Block Logic: Clears the list of `MyThread` instances for the next timepoint.
            self.device.thread_list = []

            # Block Logic: Clears the timepoint_done event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.timepoint_done.clear()

            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.device.barrier.wait()
