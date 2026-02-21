"""
@file device.py
@brief Implements a simulated distributed device using a reusable barrier and multi-threaded script execution with dynamic fine-grained locking.

This module defines the `Device`, `OneThread`, and `DeviceThread` classes,
which together simulate a node in a distributed sensing network. Each `Device`
manages its local sensor data and executes assigned scripts. A `DeviceThread`
per device coordinates the overall timepoint simulation, while `OneThread`s
are spawned to concurrently execute individual scripts. A shared `ReusableBarrierSem`
ensures global timepoint synchronization, and a dynamically created list of
`Lock` objects (`block_location`) provides fine-grained control over access
to sensor data locations.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds references to the global barrier
  and the shared list of data locks.
- `OneThread`: Worker threads spawned by `DeviceThread` to execute
  individual scripts. Each script thread acquires a lock for its target
  data location, gathers data from neighbors and itself, runs the script,
  updates data, and releases the lock.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the
  execution for each timepoint, including neighborhood discovery, spawning
  `OneThread`s, and global timepoint synchronization.
- `ReusableBarrierSem`: A shared barrier for global synchronization
  across all devices at each simulation step.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `OneThread` instances (spawned by `DeviceThread`) act as consumers,
  processing them.
- Fine-grained Locking: A shared list of `Lock` objects (`block_location`) ensures
  exclusive access to sensor data at specific locations during script execution
  across concurrent worker threads.
- Dynamic Lock Creation: Locks for data locations are created dynamically
  based on the locations found in sensor data.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


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
        self.barrier = None # Reference to the global ReusableBarrierSem
        self.thread.start() # Starts the device's main thread
        self.block_location = None # Shared list of Lock objects for fine-grained data access

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device synchronization mechanisms.

        If this device is the master (device_id 0), it initializes the global
        `ReusableBarrierSem` and dynamically creates a shared list of `Lock`
        objects (`block_location`) based on all sensor locations across
        all devices. These synchronization primitives are then propagated
        to all `Device` instances.

        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: Initializes shared synchronization resources if this is the master device (device_id 0).
        if self.device_id == 0:

            self.barrier = ReusableBarrierSem(len(devices)) # Create a single global barrier
            locations = []
            # Block Logic: Gathers all unique sensor locations from all devices to initialize locks.
            for device in devices:
                for location in device.sensor_data:
                    if location is not None:
                        if location not in locations:
                            locations.append(location)
            # Block Logic: Initializes a list of `Lock` objects, one for each unique sensor location.
            self.block_location = []
            for _ in xrange(len(locations)): # Using xrange for Python 2 compatibility, equivalent to range in Python 3
                self.block_location.append(Lock())
            # Block Logic: Propagates the initialized global barrier and list of locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.block_location = self.block_location

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
        # Block Logic: Handles script assignment and signals script availability or timepoint completion.
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

class OneThread(Thread):
    """
    @brief A worker thread responsible for executing a single script on sensor data.

    Each `OneThread` processes one assigned script, acquiring the necessary
    location-specific lock, gathering data from the local device and its neighbors,
    executing the script, and propagating the results back to the relevant devices.
    """

    def __init__(self, myid, device, location, neighbours, script):
        """
        @brief Initializes a new OneThread instance.

        @param myid: An identifier for this thread (potentially for naming or debugging).
        @param device: A reference to the parent `Device` instance.
        @param location: The data location (e.g., sensor ID) for the script.
        @param neighbours: A list of neighboring `Device` instances.
        @param script: The script object to execute.
        """
        Thread.__init__(self)
        self.myid = myid # Thread identifier
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def run(self):
        """
        @brief The main execution logic for the `OneThread`.

        Acquires the location-specific lock, gathers data from neighbors and
        the local device, executes the script, updates sensor data on affected
        devices, and then releases the lock.
        """
        # Block Logic: Acquires the location-specific lock from `self.device.block_location`
        # to ensure exclusive access to the sensor data at `self.location` during script execution.
        with self.device.block_location[self.location]:
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
            if script_data != []:

                # Functional Utility: Executes the assigned script with the collected data.
                result = self.script.run(script_data)

                # Block Logic: Propagates the script's result back to neighboring devices.
                for device in self.neighbours:
                    device.set_data(self.location, result)

                # Block Logic: Updates the current device's own sensor data with the script's result.
                self.device.set_data(self.location, result)

class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device, managing its simulation logic.

    This thread is responsible for discovering neighbors, waiting for script
    assignments, spawning `OneThread`s for concurrent script execution,
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
        assignments to complete, spawns `OneThread`s to process scripts concurrently,
        waits for all script threads to finish, and then synchronizes globally
        using the shared barrier before starting the next timepoint.
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


            # Block Logic: Spawns `OneThread` instances for each assigned script to execute them concurrently.
            threads = []
            myid = 0 # Counter for naming worker threads
            for (script, location) in self.device.scripts:
                thread = OneThread(myid, self.device, location, neighbours, script)
                threads.append(thread)
                thread.start() # Start the worker thread
                myid += 1
            # Block Logic: Waits for all `OneThread` instances to complete their execution.
            for thread in threads:
                thread.join()

            # Block Logic: Clears the timepoint_done event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.device.barrier.wait()