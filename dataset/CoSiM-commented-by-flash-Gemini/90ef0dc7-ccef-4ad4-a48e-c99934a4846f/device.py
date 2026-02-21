"""
@file device.py
@brief Implements a simulated device and its multi-threaded execution model for distributed sensing and script processing.

This module defines the `Device`, `DeviceThread`, and `WorkerThread` classes,
which together simulate a distributed system of interconnected devices.
Each `Device` manages its local sensor data and executes assigned scripts
concurrently using multiple worker threads. A master device (device_id 0)
is responsible for initializing shared synchronization primitives like
location-specific locks and a global barrier.

Architecture:
- `Device`: Represents a single node in a distributed sensing network.
  Manages local state, inter-device communication through a supervisor,
  and a queue for scripts. It also holds shared synchronization objects
  initialized by the master device.
- `DeviceThread`: A single dedicated thread per `Device` that orchestrates
  the execution for a specific timepoint. It discovers neighbors, creates
  and manages `WorkerThread` instances, and handles timepoint-level synchronization.
- `WorkerThread`: Multiple worker threads per `Device` that concurrently
  fetch scripts from a queue, acquire locks for data locations, execute scripts,
  and update sensor data on the device and its neighbors.
- `ReusableBarrierCond`: Used for synchronization across multiple devices at
  timepoint boundaries.
- `Queue`: Manages incoming scripts for processing by `WorkerThread` instances.

Patterns:
- Master-Slave: Device 0 acts as a master for setting up shared resources.
- Producer-Consumer: The `assign_script` method acts as a producer, adding scripts
  to a queue, while `WorkerThread` instances act as consumers.
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Fine-grained Locking: `location_locks` ensure exclusive access to sensor data
  at specific locations during script execution across concurrent worker threads.
"""
from threading import Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, orchestrates script execution using worker threads,
    and participates in synchronization protocols for distributed simulation.
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

        # Functional Utility: Queue for scripts assigned to this device, consumed by worker threads.
        self.queue = Queue()
        self.num_threads = 8  # Number of worker threads per device

        # Functional Utility: Shared synchronization objects, initialized by master device.
        self.location_locks = None  # Dictionary of locks for each sensor data location
        self.lock = None  # General purpose lock for critical sections within the device
        self.barrier = None  # Barrier for synchronizing all devices at timepoint end

        self.thread = None  # The main DeviceThread instance for this device

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device communication and synchronization mechanisms.

        The master device (device_id 0) initializes shared location locks and a
        global barrier. These shared resources are then propagated to all other devices.
        Each device also initializes and starts its dedicated `DeviceThread`.

        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: Master device (device_id 0) initializes shared resources.
        # Pre-condition: This block executes once at the start of the simulation by device 0.
        if self.device_id == 0:
            # Invariant: `location_locks` will contain a lock for each unique sensor data location.
            self.location_locks = {}
            # Invariant: `self.lock` is a re-entrant lock for protecting shared data structures.
            self.lock = Lock()
            # Invariant: `self.barrier` is initialized with the total number of devices,
            # ensuring all devices synchronize at each timepoint.
            self.barrier = ReusableBarrierCond(len(devices))

            # Block Logic: Propagates the shared synchronization objects to all other devices.
            # Invariant: All non-master devices receive references to the master's shared locks and barrier.
            for device in devices:
                if device.device_id != 0:
                    device.location_locks = self.location_locks
                    device.lock = self.lock
                    device.barrier = self.barrier
        # Functional Utility: Initializes and starts the main thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the device's script queue.
        A lock for the specific location is created if it doesn't already exist.
        If no script is provided (None), a shutdown signal is sent to all worker threads.

        @param script: The script object to execute, or None to signal shutdown.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """
        # Block Logic: Adds a script to the queue for processing by worker threads.
        # Pre-condition: 'script' is an executable object.
        if script is not None:
            # Block Logic: Ensures atomic creation of a location-specific lock if not present.
            with self.lock:
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()
            self.queue.put((script, location))
        # Block Logic: Signals all worker threads to terminate by sending None.
        else:
            for _ in range(self.num_threads):
                self.queue.put((None, None))

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The identifier for the data location.
        @return The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[
            location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The identifier for the data location.
        @param data: The new data to set.
        """
        # Block Logic: Updates sensor data only if the location already exists.
        # This prevents accidental creation of new data locations.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its main thread to complete.

        Functional Utility: Ensures proper termination and cleanup of resources
        associated with the device's concurrent execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device, orchestrating timepoint execution.

    This thread manages the lifecycle of worker threads, neighborhood discovery,
    and timepoint-level synchronization for its parent Device.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: A reference to the parent Device.
        """
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Continuously discovers neighbors, creates and manages WorkerThreads to
        process scripts, and synchronizes with other devices at the end of
        each timepoint until a shutdown signal is received.
        """
        while True:
            # Block Logic: Discovers neighboring devices for the current timepoint.
            # Pre-condition: `self.device.supervisor` is available to provide neighborhood information.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks for a shutdown condition (None neighbors indicates termination).
            if neighbours is None:
                break

            # Block Logic: Creates and starts multiple WorkerThreads for concurrent script processing.
            # Invariant: `self.device.num_threads` WorkerThreads are created for each timepoint.
            worker_threads = [WorkerThread(self.device, neighbours) for _ in
                              range(self.device.num_threads)]
            for thread in worker_threads:
                thread.start()
            # Block Logic: Waits for all WorkerThreads to complete their tasks for the current timepoint.
            for thread in worker_threads:
                thread.join()

            # Block Logic: Synchronizes all devices in the simulation at the end of the timepoint.
            # Invariant: All devices complete their script processing before advancing to the next timepoint.
            self.device.barrier.wait()


class WorkerThread(Thread):
    """
    @brief A worker thread responsible for fetching and executing scripts.

    Multiple instances of WorkerThread run concurrently within a Device to
    process scripts from a shared queue, applying them to local and
    neighboring sensor data.
    """

    def __init__(self, device, neighbours):
        """
        @brief Initializes a new WorkerThread instance.

        @param device: A reference to the parent Device.
        @param neighbours: A list of neighboring Device instances.
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours

    def run_script(self, script, location):
        """
        @brief Executes a given script on sensor data from the current device and its neighbors.

        Collects data for the specified location from all neighbors and the current device,
        executes the script, and then propagates the result back to all participating devices.

        @param script: The script object to be executed.
        @param location: The data location (e.g., sensor ID) to operate on.
        """
        script_data = []
        # Block Logic: Collects sensor data from neighboring devices for script input.
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        # Block Logic: Collects sensor data from the current device for script input.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script only if there is data to process.
        if script_data:
            # Functional Utility: Executes the assigned script with the collected data.
            result = script.run(script_data)

            # Block Logic: Propagates the script's result back to neighboring devices.
            for device in self.neighbours:
                device.set_data(location, result)
            # Block Logic: Updates the current device's own sensor data with the script's result.
            self.device.set_data(location, result)

    def run(self):
        """
        @brief The main execution loop for the WorkerThread.

        Continuously fetches scripts from the device's queue, acquires
        location-specific locks, executes the scripts using `run_script`,
        and then returns the processed script to the queue for potential
        re-execution in subsequent timepoints (effectively simulating a loop
        of script availability until a None script is received for shutdown).
        """
        while True:
            # Block Logic: Fetches a script and its location from the device's queue.
            # Pre-condition: `self.device.queue` contains (script, location) tuples.
            script, location = self.device.queue.get()
            # Block Logic: Terminates the worker thread if a None script (shutdown signal) is received.
            if script is None:
                return
            # Block Logic: Acquires a location-specific lock before executing the script
            # to ensure exclusive access to the sensor data at that location.
            with self.device.location_locks[location]:
                self.run_script(script, location)
            # Block Logic: Puts the script back into the queue. This pattern suggests
            # that scripts might be re-executed or are part of a continuous process
            # over multiple timepoints, or it's a mechanism for keeping the queue
            # populated for other workers.
            self.device.queue.put((script, location))
