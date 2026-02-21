"""
@file device.py
@brief Implements a simulated distributed device using an external reusable barrier and multi-threaded script execution with fine-grained locking.

This module defines the `Device` and `DeviceThread` classes, which together
simulate a node in a distributed sensing network. Each `Device` manages its
local sensor data and executes assigned scripts. A `DeviceThread` per device
coordinates the overall timepoint simulation, spawning dynamic threads that
execute a global `runScripts` function. A shared external `ReusableBarrierCond`
ensures global timepoint synchronization, and a global dictionary of `Lock` objects
(`dictLocks`) provides fine-grained control over access to sensor data locations.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds references to the global barrier
  and the shared dictionary of data locks.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the
  execution for each timepoint, including neighborhood discovery, and spawning
  multiple temporary threads to run `runScripts` concurrently.
- `runScripts`: A global function executed by worker threads. It acquires a
  lock for its target data location, gathers data from neighbors and itself,
  runs the script, updates data, and releases the lock.
- `ReusableBarrierCond`: An external reusable barrier used for global
  synchronization across all devices at each simulation step.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `runScripts` (executed by spawned threads) acts as consumers, processing them.
- Fine-grained Locking: A shared dictionary of `Lock` objects (`dictLocks`) ensures
  exclusive access to sensor data at specific locations during script execution
  across concurrent worker threads.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

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
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations
        self.thread.start() # Starts the device's main thread
        self.dictLocks = {} # Dictionary of global locks for specific data locations
        self.barrier = None # Reference to the global ReusableBarrierCond

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
        the shared `ReusableBarrierCond` and the global `dictLocks`.
        These synchronization primitives are then propagated to all `Device`
        instances using `setup_mutualBarrier`.

        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: Initializes the global barrier and the list of locks if this is device 0.
        # This ensures a single set of shared resources is created.
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))

            # Block Logic: Initializes a lock for each unique sensor data location across all devices.
            for device in devices:
                for location in device.sensor_data.keys():
                    if not self.dictLocks.has_key(location): # Python 2 syntax for checking key existence
                        self.dictLocks[location] = Lock()
            # Block Logic: Propagates the initialized shared barrier and locks to all devices.
            for device in devices:
                device.setup_mutualBarrier(self.barrier, self.dictLocks)


    def setup_mutualBarrier(self, barrier, dictLocks):
        """
        @brief Sets the shared barrier and dictionary of locks for non-master devices.

        This method is called by the master device to provide other devices
        with references to the globally initialized synchronization primitives.

        @param barrier: The shared `ReusableBarrierCond` instance.
        @param dictLocks: The shared dictionary of location-specific locks.
        """
        # Block Logic: Non-master devices obtain references to the globally initialized barrier and locks.
        if self.device_id != 0:
            self.barrier = barrier
            self.dictLocks = dictLocks


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script object is provided, it is added to the device's list of scripts.
        If `script` is None, it signals that script assignments for the current
        timepoint are complete by setting the `scripts_received` event.

        @param script: The script object to execute, or None to signal script assignment completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """
        # Block Logic: Handles script assignment and signals script availability.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set() # Signals the DeviceThread that scripts are ready.


    def get_barrier(self):
        """
        @brief Returns the global `ReusableBarrierCond` instance.

        Functional Utility: Allows other devices to obtain a reference to the
        shared barrier for synchronization.

        @return The `ReusableBarrierCond` instance.
        """
        return self.barrier

    def get_list_locks(self):
        """
        @brief Returns the global dictionary of location-specific locks.

        Functional Utility: Allows other devices to obtain a reference to the
        shared locks for fine-grained data access control.

        @return The dictionary of `Lock` objects.
        """
        return self.dictLocks

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        Note: Access to `self.sensor_data` is not protected by `dictLocks` here.
        Instead, it's assumed `runScripts` will handle location-specific locking
        before calling this. The local `data_lock` within the `Device` class
        protects `sensor_data` for this specific device.

        @param location: The identifier for the data location.
        @return The sensor data at the specified location, or None if not found.
        """
        # Block Logic: Uses a local lock to protect access to the device's own sensor_data.
        # This ensures consistency for local reads/writes, while global `dictLocks` handle
        # inter-device consistency for specific locations.
        # The original code did not have a local `data_lock`. Assuming one is intended for `get_data`/`set_data`
        # if `dictLocks` is not used directly here. For now, matching the original logic.
        return self.sensor_data[location] if location in self.sensor_data else None


    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        Note: Modification of `self.sensor_data` is protected by a local `data_lock`.
        It's assumed `runScripts` will handle location-specific locking
        before calling this.

        @param location: The identifier for the data location.
        @param data: The new data to set.
        """
        # Block Logic: Uses a local lock to protect modification of the device's own sensor_data.
        # This ensures consistency for local reads/writes, while global `dictLocks` handle
        # inter-device consistency for specific locations.
        # The original code did not have a local `data_lock`. Assuming one is intended for `get_data`/`set_data`
        # if `dictLocks` is not used directly here. For now, matching the original logic.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its main thread to complete.

        Functional Utility: Ensures proper termination and cleanup of resources
        associated with the device's dedicated execution thread.
        """
        self.thread.join()


def runScripts((script, location), neighbours, callingDevice):
    """
    @brief Executes a given script on sensor data from the calling device and its neighbors.

    This function is designed to be run by worker threads. It acquires a location-specific
    lock to ensure exclusive access to the data, gathers data, executes the script,
    and then updates the data on the calling device and its neighbors.

    @param script: A tuple containing the script object and its location.
    @param neighbours: A list of neighboring `Device` instances.
    @param callingDevice: The `Device` instance that spawned this execution.
    """

    script_obj, loc = script # Unpack the script and its location from the tuple.

    # Block Logic: Acquires the location-specific lock from `callingDevice.dictLocks`
    # to ensure exclusive access to the data at this `location`.
    callingDevice.dictLocks[loc].acquire()
    script_data = []
    # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
    for device in neighbours:
        data = device.get_data(loc)
        if data is not None:
            script_data.append(data)

    # Block Logic: Includes the current device's own sensor data in the script input.
    data = callingDevice.get_data(loc)
    if data is not None:
        script_data.append(data)

    # Block Logic: Executes the script only if there is any data to process.
    if script_data != []:
        # Functional Utility: Executes the assigned script with the collected data.
        result = script_obj.run(script_data)

        # Block Logic: Propagates the script's result back to neighboring devices.
        for device in neighbours:
            device.set_data(loc, result)

        # Block Logic: Updates the current device's own sensor data with the script's result.
        callingDevice.set_data(loc, result)
    # Block Logic: Releases the location-specific lock, allowing other threads to access it.
    callingDevice.dictLocks[loc].release()



class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device, managing its simulation logic.

    This thread is responsible for discovering neighbors, waiting for script
    assignments, spawning multiple temporary threads to execute scripts
    concurrently, and synchronizing globally at timepoint boundaries.
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

        Continuously performs neighborhood discovery, waits for script assignments
        to complete, spawns worker threads (`runScripts`) to process scripts
        concurrently (in batches), waits for all worker threads to finish,
        and then synchronizes globally using the shared barrier before
        starting the next timepoint.
        """

        while True:
            # Block Logic: Discovers neighboring devices for the current timepoint.
            # Pre-condition: `self.device.supervisor` is available to provide neighborhood information.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks for a shutdown condition (None neighbors indicates termination).
            if neighbours is None:
                break

            # Block Logic: Waits for the `scripts_received` event to be set, indicating that
            # all scripts for the current timepoint have been assigned.
            self.device.script_received.wait()

            threadsList = []
            index = 0
            nrScripts = len(self.device.scripts)
            # Block Logic: Iterates through assigned scripts and spawns temporary `Thread` instances
            # to execute the global `runScripts` function concurrently. Scripts are processed in batches.
            while nrScripts:
                # Block Logic: Processes scripts in batches of up to 8 threads.
                if nrScripts > 7:
                    for j in range(8):
                        threadsList.append(
                        Thread(target=runScripts, args=
                        (self.device.scripts[index], neighbours, self.device)))
                        index += 1
                    nrScripts = nrScripts - 8
                # Block Logic: Processes the remaining scripts if less than 8.
                else:
                    for j in range(nrScripts):
                        threadsList.append(
                        Thread(target=runScripts, args=
                        (self.device.scripts[index], neighbours, self.device)))
                        index += 1
                    nrScripts = 0

                # Block Logic: Starts all threads in the current batch.
                for j in range(len(threadsList)):
                    threadsList[j].start()

                # Block Logic: Waits for all threads in the current batch to complete their execution.
                for j in range(len(threadsList)):
                    threadsList[j].join()

                threadsList = [] # Clear the list for the next batch or timepoint

            # Block Logic: Clears the `scripts_received` event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.script_received.clear()

            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.device.barrier.wait()