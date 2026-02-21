"""
@file device.py
@brief Implements a simulated distributed device using an external reusable barrier and multi-threaded script execution with fine-grained locking and batching.

This module defines the `Device` and `DeviceThread` classes, which together
simulate a node in a distributed sensing network. Each `Device` manages its
local sensor data and processes assigned scripts. A `DeviceThread` per device
coordinates the overall timepoint simulation, dynamically spawning threads
to execute scripts in batches. A shared external `ReusableBarrierSem` ensures
global timepoint synchronization, and a global dictionary of `Lock` objects
(`location_lock`) provides fine-grained control over access to sensor data locations.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds references to the global barrier
  and the shared dictionary of data locks. It also provides a `run_script` method
  that can be executed by worker threads.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the
  execution for each timepoint, including neighborhood discovery, and dynamically
  spawning worker threads to run scripts concurrently in batches.
- `ReusableBarrierSem`: An external reusable barrier used for global
  synchronization across all devices at each simulation step.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  Worker threads spawned by `DeviceThread` (executing `Device.run_script`)
  act as consumers, processing them.
- Fine-grained Locking: A shared dictionary of `Lock` objects (`location_lock`) ensures
  exclusive access to sensor data at specific locations during script execution.
- Script Batching: Worker threads are managed in batches (e.g., up to 8 concurrent threads)
  to control resource utilization.
"""

from threading import Event, Thread, Lock, Condition


class ReusableBarrier():
    """
    @brief A reusable barrier that synchronizes a fixed number of threads using a Condition variable.

    This barrier allows a specified number of threads to wait until all
    have arrived. Once all threads arrive, they are all released, and
    the barrier resets for subsequent reuse.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier instance.

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
        self.scripts = [] # List to store assigned scripts
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations
        self.thread.start() # Starts the device's main thread

        self.timepoint_done = Event() # Event to signal that timepoint script assignment is complete
        self.script_received = Event() # Event to signal that scripts are ready to be processed
        self.barrier = None # Reference to the global ReusableBarrier
        self.location_lock = None # Reference to the shared dictionary of locks for data locations
        self.lock = Lock() # Local lock for this device's sensor_data methods

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
        `ReusableBarrier` for all devices and the shared `location_lock` dictionary.
        Other devices obtain references to these shared resources.

        @param devices: A list of all Device instances in the simulation.
        """

        # Block Logic: Initializes the global barrier and the location_lock dictionary if this is device 0.
        # This ensures a single set of shared resources is created and distributed.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices)) # Create a single global barrier
            self.location_lock = {} # Initialize the global dictionary of locks
            # Block Logic: Populates the location_lock dictionary with locks for all known sensor locations.
            for device in devices:
                for location in device.sensor_data:
                    self.location_lock[location] = Lock() # Initialize a Lock for each unique location
        # Block Logic: Non-master devices obtain references to the globally initialized barrier and locks.
        else:
            self.barrier = devices[0].barrier
            self.location_lock = devices[0].location_lock


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

        Access to `self.sensor_data` is protected by the local `self.lock`.

        @param location: The identifier for the data location.
        @return The sensor data at the specified location, or None if not found.
        """
        with self.lock: # Acquire local lock for reading sensor_data
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        Modification of `self.sensor_data` is protected by the local `self.lock`.

        @param location: The identifier for the data location.
        @param data: The new data to set.
        """
        with self.lock: # Acquire local lock for writing sensor_data
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its main thread to complete.

        Functional Utility: Ensures proper termination and cleanup of resources
        associated with the device's dedicated execution thread.
        """
        self.thread.join()

    def run_script(self, script, location, neighbours):
        """
        @brief Executes a given script on sensor data from the calling device and its neighbors.

        This method is designed to be executed by worker threads. It acquires a location-specific
        lock to ensure exclusive access to the data, gathers data, executes the script,
        and then updates the data on the calling device and its neighbors.

        @param script: The script object to be executed.
        @param location: The data location (e.g., sensor ID) to operate on.
        @param neighbours: A list of neighboring `Device` instances.
        """
        # Block Logic: Acquires the location-specific lock from `self.location_lock`
        # to ensure exclusive access to the data at this `location`.
        self.location_lock[location].acquire()
        script_data = []

        # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Includes the current device's own sensor data in the script input.
        data = self.get_data(location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script only if there is any data to process.
        if script_data != []:

            # Functional Utility: Executes the assigned script with the collected data.
            result = script.run(script_data)

            # Block Logic: Propagates the script's result back to neighboring devices.
            for device in neighbours:
                device.set_data(location, result)
            # Block Logic: Updates the current device's own sensor data with the script's result.
            self.set_data(location, result)

        # Block Logic: Releases the location-specific lock, allowing other threads to access it.
        self.location_lock[location].release()


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

        Continuously performs neighborhood discovery, waits for timepoint script
        assignments to complete, spawns worker threads (executing `Device.run_script`)
        to process scripts concurrently (in batches of 8), waits for all worker
        threads to finish, and then synchronizes globally using the shared barrier
        before starting the next timepoint.
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


            self.device.timepoint_done.clear() # Clear the event for the next timepoint.

            threads = [] # List to hold worker threads for the current batch
            script_index = 0
            # Block Logic: Processes scripts in batches of 8, spawning worker threads for concurrent execution.
            # Invariant: Up to 8 scripts are processed concurrently before waiting for their completion.
            while script_index < len(self.device.scripts):
                script, location = self.device.scripts[script_index]
                script_index += 1

                t = Thread(target=self.device.run_script, args=(script, location, neighbours))
                t.start()
                threads.append(t)

                # Block Logic: If 8 threads are running or all scripts are assigned, wait for them to finish.
                if len(threads) == 8 or script_index == len(self.device.scripts):
                    for thread in threads:
                        thread.join()
                    threads = [] # Clear the list for the next batch.


            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.device.barrier.wait()