"""
@file device.py
@brief Implements a simulated distributed device using a custom reusable barrier and multi-threaded script execution with per-device locking.

This module defines the `ReusableBarrier`, `MyThread`, `Device`, and `DeviceThread` classes,
which together simulate a node in a distributed sensing network. Each `Device`
manages its local sensor data and executes assigned scripts. A `DeviceThread`
per device coordinates the overall timepoint simulation, while `MyThread`s
are spawned to concurrently execute individual scripts. A shared `ReusableBarrier`
ensures global timepoint synchronization, and each `Device` instance has its
own `Lock` (`self.lock`) to protect its sensor data during updates.

Architecture:
- `ReusableBarrier`: A custom implementation of a reusable barrier that uses a
  `threading.Condition` to synchronize multiple threads/devices.
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds a reference to the global barrier
  and a local lock for its data.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the
  execution for each timepoint, including neighborhood discovery, spawning
  `MyThread`s, and global timepoint synchronization.
- `MyThread`: Worker threads spawned by `DeviceThread` to execute
  individual scripts. Each script thread gathers data from neighbors and itself,
  runs the script, and updates data, acquiring and releasing locks as necessary.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `MyThread` instances (spawned by `DeviceThread`) act as consumers,
  processing them.
- Per-Device Locking: Each `Device` has its own lock (`self.lock`) to protect
  access to its `sensor_data` when updated by `MyThread`s.
"""

from threading import Event, Thread, Condition, Lock


class ReusableBarrier():
    """
    @brief A reusable barrier that synchronizes a fixed number of threads using a Condition variable.

    This barrier allows a specified number of threads to wait until all
    have arrived. Once all threads arrive, they are all released, and
    the barrier resets for subsequent reuse.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrierCond.

        @param num_threads: The total number of threads that must arrive
                            at the barrier before it releases.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Current count of threads waiting

        self.cond = Condition() # Condition variable for synchronization

    def wait(self):
        """
        @brief Blocks until all `num_threads` have called this method.

        When the last thread arrives, all waiting threads are notified and released.
        The barrier then resets its internal count, becoming ready for reuse.
        """
        self.cond.acquire() # Acquire the lock associated with the Condition

        self.count_threads -= 1
        # Block Logic: If this is the last thread to arrive, notify all others and reset.
        if self.count_threads == 0:
            self.cond.notify_all() # Release all waiting threads

            self.count_threads = self.num_threads # Reset for next use

        # Block Logic: If not the last thread, wait until notified by the last thread.
        else:
            self.cond.wait() # Release the lock and wait; reacquires lock on wakeup

        self.cond.release() # Release the lock

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
        self.thread.start() # Starts the device's main thread
        self.barrier = None # Reference to the global ReusableBarrier
        self.lock = Lock() # Local lock for this device's sensor_data

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization resources among devices.

        If this device is the first (device_id 0), it initializes the global
        `ReusableBarrier` for all devices. Other devices obtain a reference
        to this shared barrier.

        @param devices: A list of all Device instances in the simulation.
        """

        # Block Logic: Initializes the global barrier instance if this is device 0,
        # otherwise, it obtains a reference to the barrier initialized by device 0.
        # This ensures a single barrier is shared across all devices.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        else:
            self.barrier = devices[0].barrier


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script object is provided, it is added to the device's list of scripts
        to be processed, and `script_received` event is set. If `script` is None,
        it signals that all scripts for the current timepoint have been assigned
        by setting the `timepoint_done` event.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """
        # Block Logic: Differentiates between assigning a script and signaling end-of-assignment.
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

    Each `MyThread` processes one assigned script, gathers necessary
    sensor data from the local device and its neighbors, executes the script,
    and updates the results back to the relevant devices, acquiring and
    releasing locks on each device for data consistency.
    """

    def __init__(self, neighbours, device, location, script):
        """
        @brief Initializes a new MyThread instance.

        @param neighbours: A list of neighboring `Device` instances.
        @param device: A reference to the parent `Device` instance.
        @param location: The data location (e.g., sensor ID) for the script.
        @param script: The script object to execute.
        """
        Thread.__init__(self)


        self.neighbours = neighbours
        self.device = device
        self.location = location
        self.script = script

    def run(self):
        """
        @brief The main execution logic for the `MyThread`.

        Gathers data from neighbors and the local device, executes the script,
        and updates the sensor data on affected devices, using per-device locks
        to ensure data integrity during updates.
        """
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
            # Block Logic: Propagates the script's result back to neighboring devices,
            # acquiring and releasing their individual locks for thread-safe updates.
            for device in self.neighbours:
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()

            # Block Logic: Updates the current device's own sensor data with the script's result,
            # acquiring and releasing its local lock for thread-safe updates.
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()

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

            # Block Logic: Clears the `timepoint_done` event, resetting it for the next timepoint.
            # This is done *before* starting threads to ensure it's clear for the next iteration.
            self.device.timepoint_done.clear()

            threads = []
            # Block Logic: Spawns a `MyThread` for each assigned script to execute them concurrently.
            # Pre-condition: `self.device.scripts` contains all scripts to be processed in this timepoint.
            for (script, location) in self.device.scripts:
                t = MyThread(neighbours, self.device, location, script)
                t.start()
                threads.append(t)

            # Block Logic: Waits for all `MyThread` instances to complete their execution.
            for i in range(len(threads)):
                threads[i].join()

            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.device.barrier.wait()
