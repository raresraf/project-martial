"""
@file device.py
@brief Implements a simulated distributed device using a reusable barrier and multi-threaded script execution with fine-grained locking.

This module defines the `Device`, `DeviceThread`, and `ScriptThread` classes,
which together simulate a node in a distributed sensing network. Each `Device`
manages its local sensor data and executes assigned scripts. A `DeviceThread`
per device coordinates the overall timepoint simulation, while `ScriptThread`s
are spawned to concurrently execute individual scripts. A shared `ReusableBarrier`
ensures global timepoint synchronization, and location-specific `Lock` objects
provide fine-grained control over data access.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds a reference to the global barrier.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the
  execution for each timepoint, including neighborhood discovery, spawning
  `ScriptThread`s, and global timepoint synchronization. It also manages
  a dictionary of location-specific locks.
- `ScriptThread`: Worker threads spawned by `DeviceThread` to execute
  individual scripts. Each script thread acquires a lock for its target
  data location, gathers data from neighbors and itself, runs the script,
  updates data, and releases the lock.
- `barrier.ReusableBarrier`: A shared barrier for global synchronization
  across all devices at each simulation step.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `ScriptThread` instances (spawned by `DeviceThread`) act as consumers,
  processing them.
- Fine-grained Locking: `locations_lock` ensures exclusive access to sensor data
  at specific locations during script execution across concurrent worker threads.
"""

from threading import Event, Thread, Lock
import barrier

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
        self.devices = None # Reference to the list of all devices, set in setup_devices
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to store assigned scripts
        self.timepoint_done = Event() # Event to signal that timepoint script assignment is complete
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations
        self.thread.start() # Starts the device's main thread

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device synchronization mechanisms.

        Initializes a shared `ReusableBarrier` for all devices and propagates
        it to each device's thread. It also stores a reference to all devices.
        This method assumes `devices` is a list of all `Device` objects.

        @param devices: A list of all Device instances in the simulation.
        """

        # Functional Utility: Creates a single shared barrier instance for all devices.
        shared_barrier = barrier.ReusableBarrier(len(devices))

        # Block Logic: Propagates the shared barrier to each device's thread.
        # This ensures all DeviceThreads synchronize using the same barrier.
        if self.device_id == 0: # Assuming device_id 0 is responsible for initialization
            for i in xrange(len(devices)):
                devices[i].thread.barrier = shared_barrier

        # Functional Utility: Stores a reference to all devices for inter-device communication.
        self.devices = devices

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the device's script list.
        A location-specific lock is created if it doesn't already exist and
        then propagated to all device threads for shared access control.
        If no script is provided (None), it signals that script assignments
        for the current timepoint are complete by setting the `timepoint_done` event.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """
        # Block Logic: Handles script assignment and ensures location locks are correctly setup.
        if script is not None:
            self.scripts.append((script, location))
            # Block Logic: Initializes a location-specific lock if it doesn't exist.
            # This lock is then shared among all device threads to protect access to this location.
            if location not in self.thread.locations_lock:
                loc_lock = Lock()
                # Functional Utility: Propagates the newly created lock to all device threads.
                for i in xrange(len(self.devices)):
                    self.devices[i].thread.locations_lock[location] = loc_lock

        else:
            self.timepoint_done.set() # Signals the DeviceThread that script assignments are complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The identifier for the data location.
        @return The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] \
        if location in self.sensor_data else None

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

    This thread is responsible for discovering neighbors, waiting for script
    assignments, spawning `ScriptThread`s for concurrent script execution,
    and synchronizing globally at timepoint boundaries.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: A reference to the parent Device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = None # Reference to the shared global barrier, set in Device.setup_devices
        self.script_threads = [] # List to hold ScriptThread instances for the current timepoint
        self.locations_lock = {} # Dictionary of locks for specific data locations

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Continuously performs neighborhood discovery, waits for timepoint script
        assignments to complete, spawns `ScriptThread`s to process scripts concurrently,
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

            count = 0
            # Block Logic: Iterates through assigned scripts and spawns `ScriptThread`s for concurrent execution.
            # Pre-condition: `self.device.scripts` contains all scripts to be processed in this timepoint.
            for (script, location) in self.device.scripts:
                # Block Logic: Limits the number of concurrently running script threads to 8.
                # It joins completed threads before starting new ones to manage resources.
                if count == 8:
                    count = 0
                    for i in xrange(len(self.script_threads)):
                        self.script_threads[i].join()
                    del self.script_threads[:] # Clear the list of joined threads

                script_thread = ScriptThread(self.device, script, location,\
                    neighbours, count, self.locations_lock) # Create a new ScriptThread

                self.script_threads.append(script_thread)
                script_thread.start() # Start the script thread
                count = count + 1

            # Block Logic: Waits for all remaining script threads to complete their execution.
            for i in xrange(len(self.script_threads)):
                self.script_threads[i].join()

            # Block Logic: Clears the timepoint_done event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.barrier.wait()

class ScriptThread(Thread):
    """
    @brief A worker thread responsible for executing a single script on sensor data.

    Each `ScriptThread` processes one assigned script, acquiring the necessary
    location-specific lock, gathering data from the local device and its neighbors,
    executing the script, and propagating the results back to the relevant devices.
    """

    def __init__(self, device, script, location, neighbours, i, locations_lock):
        """
        @brief Initializes a new ScriptThread instance.

        @param device: A reference to the parent `Device` instance.
        @param script: The script object to execute.
        @param location: The data location (e.g., sensor ID) for the script.
        @param neighbours: A list of neighboring `Device` instances.
        @param i: An index, likely for naming the thread.
        @param locations_lock: A dictionary of locks for specific data locations.
        """
        Thread.__init__(self, name="Script Thread %d%d" % (device.device_id, i))
        self.device = device


        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.locations_lock = locations_lock


    def run(self):
        """
        @brief The main execution logic for the `ScriptThread`.

        Acquires the location-specific lock, gathers data from neighbors and
        the local device, executes the script, updates sensor data on affected
        devices, and then releases the lock.
        """
        script_data = []

        # Block Logic: Acquires the location-specific lock to ensure exclusive access
        # to the sensor data at `self.location` during script execution.
        self.locations_lock[self.location].acquire()

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

        # Block Logic: Releases the location-specific lock, allowing other threads to access it.
        self.locations_lock[self.location].release()
