"""
@file device.py
@brief Implements a simulated distributed device with master and worker threads for concurrent script execution.

This module defines the `Device` class, which simulates a node in a distributed
sensing network. Each `Device` manages its local sensor data and uses a master
thread to orchestrate the execution of assigned scripts via multiple worker
threads. Synchronization across devices at timepoint boundaries is handled
by a shared barrier, and fine-grained locking ensures safe concurrent access
to sensor data.

Architecture:
- `Device`: Represents a single node. Contains a master thread and manages
  sensor data, scripts, and synchronization primitives.
- `master_func`: The main logic for the device's master thread. It discovers
  neighbors, waits for scripts, dispatches them to worker threads, and
  synchronizes with other devices.
- `worker_func`: The logic for worker threads. Each worker processes a single
  script, acquiring necessary data locks, executing the script, and updating
  sensor data on itself and its neighbors.
- `RLock`: Used for protecting access to specific sensor data locations.
- `Semaphore`: Controls the maximum number of active worker threads.
- `ReusableBarrierSem`: A barrier for global synchronization across all devices
  at each simulation step.

Patterns:
- Master-Worker: A master thread coordinates tasks and worker threads perform
  the actual script execution.
- Barrier Synchronization: Ensures all devices complete a simulation step
  before proceeding to the next.
- Producer-Consumer: `assign_script` acts as a producer, and `worker_func`
  consumes the scripts.
- Fine-grained Locking: Ensures data consistency when multiple workers
  access shared sensor data.
"""

from threading import Event, Thread, RLock, Semaphore
from barrier import ReusableBarrierSem

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, orchestrates script execution via a master thread
    and worker threads, and participates in global synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor, max_workers=8):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing initial sensor readings
                            for various locations.
        @param supervisor: A reference to the central supervisor managing
                           the distributed system.
        @param max_workers: The maximum number of worker threads to run concurrently.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()  # Event to signal when scripts are ready
        self.scripts = []  # List to store assigned scripts
        self.master = Thread(target=self.master_func) # Main thread for the device's operations

        self.master.start() # Starts the master thread immediately
        self.active_workers = Semaphore(max_workers) # Controls concurrent worker execution

        self.root_device = None # Reference to the master device (device_id 0) for shared resources

        self.step_barrier = None # Barrier for synchronizing all devices
        self.data_locks = {} # Dictionary of RLocks for each sensor data location

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device communication and synchronization mechanisms.

        Identifies the root device (device_id 0) and, if this is the root device,
        initializes the global step barrier and data locks for all sensor locations.
        These shared resources are then accessed by all other devices via the root device.

        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: Finds and stores a reference to the root device (device_id 0).
        for dev in devices:
            if dev.device_id == 0:
                self.root_device = dev

        # Block Logic: If this is the root device, initialize shared synchronization objects.
        # Pre-condition: This block executes only on the device with device_id 0.
        if self.device_id == 0:
            # Functional Utility: Initializes a reusable barrier to synchronize all devices
            # at the end of each simulation step.
            self.step_barrier = ReusableBarrierSem(len(devices))

            # Block Logic: Initializes a re-entrant lock for each unique sensor data location.
            # These locks ensure safe concurrent access to data from multiple worker threads
            # across different devices.
            for device in devices:
                for (location, _) in device.sensor_data.iteritems():
                    if location not in self.data_locks:
                        self.data_locks[location] = RLock()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's appended to the device's script list.
        If no script is provided (None), it signals that scripts for the current
        timepoint are fully received, triggering the master thread to proceed.

        @param script: The script object to execute, or None to signal script reception completion.
        @param location: The data location (e.g., sensor ID) where the script should operate.
        """
        # Block Logic: Appends a new script and its location to the list.
        if script is not None:
            self.scripts.append((script, location))
        # Block Logic: Signals the master thread that all scripts for the current
        # timepoint have been assigned.
        else:
            self.scripts_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location, ensuring thread-safe access.

        Accesses the sensor data while holding the appropriate data lock managed
        by the root device.

        @param location: The identifier for the data location.
        @return The sensor data at the specified location, or None if not found.
        """
        # Block Logic: Acquires the RLock for the specific location to ensure
        # exclusive (or shared-read) access to the sensor data.
        with self.root_device.data_locks[location]:
            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location, ensuring thread-safe access.

        Modifies the sensor data while holding the appropriate data lock managed
        by the root device.

        @param location: The identifier for the data location.
        @param data: The new data to set.
        """
        # Block Logic: Acquires the RLock for the specific location to ensure
        # exclusive write access to the sensor data.
        with self.root_device.data_locks[location]:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its master thread to complete.

        Functional Utility: Ensures proper termination and cleanup of resources
        associated with the device's concurrent execution.
        """
        self.master.join()

    def master_func(self):
        """
        @brief The main loop for the device's master thread.

        This thread continuously discovers neighbors, waits for scripts to be assigned,
        dispatches them to worker threads, and then waits for all workers to complete
        before synchronizing with other devices globally.
        """
        while True:
            # Block Logic: Discovers neighboring devices for the current timepoint.
            # Pre-condition: `self.supervisor` is available to provide neighborhood information.
            neighbours = self.supervisor.get_neighbours()
            # Block Logic: Checks for a shutdown condition (None neighbors indicates termination).
            if neighbours is None:
                break

            # Block Logic: Waits until all scripts for the current timepoint have been assigned.
            # Invariant: `self.scripts` contains all scripts for the current timepoint after this wait.
            self.scripts_received.wait()

            workers = []
            # Block Logic: Iterates through assigned scripts, creating and starting a worker thread for each.
            # Invariant: The number of concurrently active worker threads is limited by `self.active_workers` semaphore.
            for (script, location) in self.scripts:
                # Block Logic: Acquires a permit from the semaphore, potentially blocking
                # if the maximum number of active workers has been reached.
                self.active_workers.acquire()

                # Functional Utility: Creates a new thread to execute the `worker_func` for each script.
                worker = Thread(target=self.worker_func, \
                    args=(script, location, neighbours))
                workers.append(worker)
                worker.start()

            # Block Logic: Waits for all worker threads dispatched in this timepoint to complete their execution.
            for worker in workers:
                worker.join()

            # Block Logic: Resets the scripts_received event for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.scripts_received.clear()
            # Block Logic: Synchronizes all devices at the end of the simulation step.
            # Invariant: All devices have completed their script processing for the current timepoint
            # before advancing to the next.
            self.root_device.step_barrier.wait()


    def worker_func(self, script, location, neighbours):
        """
        @brief The main logic for a worker thread.

        A worker thread processes a single script for a given location,
        collecting data from neighbors and itself, executing the script,
        and updating the sensor data. All data access is protected by
        location-specific re-entrant locks.

        @param script: The script object to be executed.
        @param location: The data location (e.g., sensor ID) to operate on.
        @param neighbours: A list of neighboring Device instances.
        """
        # Block Logic: Acquires the RLock for the specific location to ensure
        # exclusive write access to the sensor data during script execution.
        with self.root_device.data_locks[location]:
            script_data = []
            # Block Logic: Collects sensor data from neighboring devices for script input.
            for dev in neighbours:
                data = dev.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: Collects sensor data from the current device for script input.
            data = self.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script only if there is data to process.
            if script_data != []:
                # Functional Utility: Executes the assigned script with the collected data.
                result = script.run(script_data)

                # Block Logic: Propagates the script's result back to neighboring devices.
                for dev in neighbours:
                    dev.set_data(location, result)

                # Block Logic: Updates the current device's own sensor data with the script's result.
                self.set_data(location, result)

        # Block Logic: Releases a permit to the semaphore, indicating that this worker
        # has completed its task and another worker can now start.
        self.active_workers.release()
