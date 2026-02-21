"""
@file device.py
@brief Implements a simulated distributed device using a reusable barrier, thread pooling, and fine-grained locking for concurrent script execution.

This module defines the `Device`, `DeviceThread`, and `ScriptWorkerThread` classes.
Together, they simulate a node in a distributed sensing network. Each `Device`
manages its local sensor data and processes assigned scripts. A `DeviceThread`
per device coordinates the overall timepoint simulation, dynamically spawning
`ScriptWorkerThread`s to process scripts concurrently. A shared `ReusableBarrier`
ensures global timepoint synchronization, a `BoundedSemaphore` manages a thread
pool for script execution, and a class-level dictionary of `Lock` objects provides
fine-grained control over access to sensor data locations.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds a `BoundedSemaphore` for thread
  pooling.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the
  execution for each timepoint, including neighborhood discovery and spawning
  `ScriptWorkerThread`s for concurrent script processing.
- `ScriptWorkerThread`: Worker threads spawned by `DeviceThread` to execute
  individual scripts. Each thread acquires a location-specific lock, gathers
  data from neighbors and itself, runs the script, updates data, and then
  releases both the data lock and a permit to the thread pool semaphore.
- `ReusableBarrier`: A shared barrier for global synchronization across all devices
  at each simulation step.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations. Implemented with a
  class-level `Device.timepoint_barrier`.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `ScriptWorkerThread` instances act as consumers, processing them.
- Thread Pooling: `BoundedSemaphore` (`max_threads_semaphore`) limits the number
  of concurrently active `ScriptWorkerThread`s, managing resource utilization.
- Fine-grained Locking: A class-level shared dictionary of `Lock` objects
  (`ScriptWorkerThread.locations_lock`) ensures exclusive access to sensor data
  at specific locations during script execution across concurrent worker threads.
"""

from threading import Event, Thread, Lock, BoundedSemaphore
from barrier import ReusableBarrier


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, processes assigned scripts concurrently,
    and participates in global synchronization with other devices.
    """

    # Functional Utility: A class-level ReusableBarrier for global timepoint synchronization,
    # initialized and protected by a lock.
    timepoint_barrier = None
    barrier_lock = Lock()

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

        self.max_threads_semaphore = BoundedSemaphore(8) # Semaphore to limit concurrent script execution to 8 threads

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    @staticmethod
    def setup_devices(devices):
        """
        @brief Static method to set up global synchronization resources.

        This method ensures that a single `ReusableBarrier` (`timepoint_barrier`)
        is initialized globally. It uses a `barrier_lock` to protect the
        initialization of the barrier, ensuring it's created only once.

        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: Double-checked locking pattern to ensure `timepoint_barrier` is initialized only once.
        if Device.timepoint_barrier is None:
            Device.barrier_lock.acquire() # Acquire lock to safely initialize barrier

            # Invariant: After this check, if barrier is still None, it means no other thread initialized it.
            if Device.timepoint_barrier is None:
                Device.timepoint_barrier = ReusableBarrier(len(devices)) # Initialize the global barrier
            Device.barrier_lock.release() # Release lock


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


class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device, managing its simulation logic.

    This thread is responsible for discovering neighbors, waiting for script
    assignments, spawning `ScriptWorkerThread`s for concurrent script execution
    (managed by a semaphore), and synchronizing globally at timepoint boundaries.
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
        assignments to complete, spawns `ScriptWorkerThread`s to process scripts
        concurrently (respecting the `max_threads_semaphore` limit), waits for all
        worker threads to finish, and then synchronizes globally using the shared
        `timepoint_barrier` before starting the next timepoint.
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

            threads = []
            # Block Logic: Iterates through all assigned scripts and dispatches them to worker threads.
            # The `max_threads_semaphore` ensures that only a limited number of threads run concurrently.
            for (script, location) in self.device.scripts:
                # Block Logic: Acquires a permit from the `max_threads_semaphore`, potentially blocking
                # if the maximum number of active worker threads (8) has been reached.
                self.device.max_threads_semaphore.acquire()

                # Functional Utility: Creates and starts a `ScriptWorkerThread` for each script.
                worker_thread = ScriptWorkerThread(self.device, neighbours, location, script)
                threads.append(worker_thread)
                worker_thread.start()

            # Block Logic: Waits for all spawned `ScriptWorkerThread`s to complete their execution.
            for thread in threads:
                thread.join()

            # Block Logic: Clears the `timepoint_done` event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.timepoint_done.clear()

            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            Device.timepoint_barrier.wait()


class ScriptWorkerThread(Thread):
    """
    @brief A worker thread responsible for executing a single script on sensor data.

    Each `ScriptWorkerThread` processes one assigned script, acquiring the necessary
    location-specific lock, gathering data from the local device and its neighbors,
    executing the script, and propagating the results back to the relevant devices.
    Finally, it releases a permit to the thread pool semaphore.
    """

    # Functional Utility: A class-level dictionary of locks for fine-grained synchronization
    # on individual sensor data locations. This ensures global consistency for each location.
    locations_lock = {}

    def __init__(self, device, neighbours, location, script):
        """
        @brief Initializes a new ScriptWorkerThread instance.

        @param device: A reference to the parent `Device` instance.
        @param neighbours: A list of neighboring `Device` instances.
        @param location: The data location (e.g., sensor ID) for the script.
        @param script: The script object to execute.
        """
        super(ScriptWorkerThread, self).__init__() # Calls the constructor of the base Thread class
        self.device = device
        self.neighbours = neighbours
        self.location = location
        self.script = script

        # Block Logic: Initializes a Lock for the given location if it doesn't already exist in the shared dictionary.
        # This ensures that each unique location has a dedicated lock.
        if location not in ScriptWorkerThread.locations_lock:
            ScriptWorkerThread.locations_lock[location] = Lock()

    def run(self):
        """
        @brief The main execution logic for the `ScriptWorkerThread`.

        Acquires the location-specific lock, gathers data from neighbors and
        the local device, executes the script, updates sensor data on affected
        devices, and then releases the lock and a permit to the thread pool semaphore.
        """

        # Block Logic: Acquires the location-specific lock to ensure exclusive access
        # to the sensor data at `self.location` during script execution.
        ScriptWorkerThread.locations_lock[self.location].acquire()

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
        ScriptWorkerThread.locations_lock[self.location].release()

        # Functional Utility: Releases a permit to the `max_threads_semaphore` in the parent `Device`,
        # indicating that a worker slot is now available for another script.
        self.device.max_threads_semaphore.release()