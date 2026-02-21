"""
@file device.py
@brief Implements a simulated distributed device using a custom barrier and multi-threaded computation for script execution.

This module defines the `ReusableBarrierCond`, `Device`, `ComputationThread`,
and `DeviceThread` classes. The `Device` class represents a node in a simulated
sensing network, managing local sensor data and coordinating script execution.
A custom `ReusableBarrierCond` provides synchronization for all devices, while
`ComputationThread` instances perform the actual script processing concurrently.

Architecture:
- `ReusableBarrierCond`: A custom barrier implementation using `threading.Condition`
  for synchronizing multiple threads/devices. It ensures all participants reach
  a specific point before proceeding.
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It also holds a reference to the global barrier.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the execution
  for each timepoint, including neighborhood discovery and spawning `ComputationThread`s.
- `ComputationThread`: Worker threads spawned by `DeviceThread` to execute
  individual scripts, gather data from neighbors, and update sensor readings.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `ComputationThread` instances (spawned by `DeviceThread`) act as consumers,
  processing them.
- Concurrent Execution: Multiple `ComputationThread` instances run in parallel
  within each device to speed up script processing.
"""

from threading import Event, Thread, Lock, Condition


class ReusableBarrierCond:
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

    # Functional Utility: A class-level barrier instance shared by all devices.
    # This design implies that the barrier is a global resource.
    barrier = None

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
        self.script_received = Event() # Event to signal when scripts are assigned
        self.scripts = [] # List to store assigned scripts
        self.timepoint_done = Event() # Event to signal that script assignment for timepoint is complete
        self.set_data_lock = Lock() # Lock to protect concurrent writes to sensor_data
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
        @brief Configures shared synchronization resources among devices.

        If this device is the first (device_id 0), it initializes the global
        `ReusableBarrierCond` for all devices. Other devices obtain a reference
        to this shared barrier.

        @param devices: A list of all Device instances in the simulation.
        """

        # Block Logic: Initializes the global barrier instance if this is device 0,
        # otherwise, it obtains a reference to the barrier initialized by device 0.
        # This ensures a single barrier is shared across all devices.
        for device in devices:
            if device.device_id == 0:
                Device.barrier = ReusableBarrierCond(len(devices))
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

        Ensures thread-safe updates to `sensor_data` using a dedicated lock.

        @param location: The identifier for the data location.
        @param data: The new data to set.
        """
        # Block Logic: Uses a lock to ensure atomic updates to the sensor_data dictionary.
        with self.set_data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its main thread to complete.

        Functional Utility: Ensures proper termination and cleanup of resources
        associated with the device's dedicated execution thread.
        """
        self.thread.join()


class ComputationThread(Thread):
    """
    @brief A worker thread responsible for executing a single script.

    Each `ComputationThread` processes one assigned script, gathers necessary
    sensor data from the local device and its neighbors, executes the script,
    and propagates the results back to the relevant devices.
    """

    def __init__(self, device_thread, neighbours, script_data):
        """
        @brief Initializes a new ComputationThread instance.

        @param device_thread: A reference to the parent `DeviceThread` instance.
        @param neighbours: A list of neighboring `Device` instances.
        @param script_data: A tuple containing the script object and its target location.
        """
        Thread.__init__(self, name="Worker %s" % device_thread.name)
        self.device_thread = device_thread
        self.neighbours = neighbours
        self.script = script_data[0] # The script to execute
        self.location = script_data[1] # The sensor data location for the script

    def run(self):
        """
        @brief The main execution logic for the `ComputationThread`.

        Gathers data from neighbors and the local device, executes the script,
        and updates the sensor data on affected devices.
        """
        script_data = []
        # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Includes the current device's own sensor data in the script input.
        data = self.device_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script only if there is data to process.
        if script_data:
            # Functional Utility: Executes the assigned script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Propagates the script's result back to neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)

            # Block Logic: Updates the current device's own sensor data with the script's result.
            self.device_thread.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device, managing its simulation logic.

    This thread is responsible for discovering neighbors, spawning
    `ComputationThread`s for script execution, and synchronizing with
    other devices at timepoint boundaries.
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
        to complete for the current timepoint, dispatches `ComputationThread`s
        to execute scripts, and then uses the global barrier to synchronize
        with all other devices before starting the next timepoint.
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

            local_threads = []
            # Block Logic: Spawns a `ComputationThread` for each assigned script.
            # Invariant: Each script is processed by an independent worker thread.
            for script_data in self.device.scripts:
                worker = ComputationThread(self, neighbours, script_data)
                worker.start()
                local_threads.append(worker)

            # Block Logic: Waits for all `ComputationThread`s spawned for the current
            # timepoint to complete their execution.
            for worker in local_threads:
                worker.join()

            # Block Logic: Clears the timepoint_done event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.timepoint_done.clear()

            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            Device.barrier.wait()
