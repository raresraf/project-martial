"""
@file device.py
@brief Implements a simulated device for distributed computation.

This file defines the `Device` class, representing a computational node
in a distributed system, and `DeviceThread`, which manages the execution
logic for each device. Devices can hold sensor data, process scripts
collaboratively with neighbors, and synchronize using barriers.
It's designed to simulate a distributed sensing and processing network.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    @class ReusableBarrier
    @brief Implements a reusable barrier synchronization mechanism.

    This barrier allows a specified number of threads (`num_threads`) to wait
    at a synchronization point and then proceed together. It's designed to be
    reused across multiple synchronization phases.

    @attribute num_threads (int): The total number of threads that must reach the barrier.
    @attribute count_threads1 (list): Internal counter for the first phase of the barrier.
    @attribute count_threads2 (list): Internal counter for the second phase of the barrier.
    @attribute count_lock (Lock): A lock to protect the internal thread counters.
    @attribute threads_sem1 (Semaphore): Semaphore for signaling threads in the first phase.
    @attribute threads_sem2 (Semaphore): Semaphore for signaling threads in the second phase.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrier instance.

        Sets up the barrier with the specified number of threads and
        initializes internal counters and semaphores for synchronization.

        @param num_threads (int): The number of threads expected to synchronize at this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier.

        A thread calling this method will block until all `num_threads`
        have also called `wait`. The barrier then releases all waiting
        threads. This method uses a two-phase approach for reusability.
        """
        # Block Logic: First phase of barrier synchronization.
        # Threads wait here until all threads have reached this point.
        self.phase(self.count_threads1, self.threads_sem1)
        # Block Logic: Second phase of barrier synchronization.
        # This phase ensures the barrier can be reused by resetting counts.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single phase of the barrier synchronization.

        This internal method handles the logic for counting threads,
        releasing them when all have arrived, and resetting the counter
        for the next use.

        @param count_threads (list): A list containing the current count of waiting threads for this phase.
        @param threads_sem (Semaphore): The semaphore used to block and release threads for this phase.
        """
        # Block Logic: Protects access to the shared `count_threads` variable.
        with self.count_lock:
            # Inline: Decrement the count of threads yet to reach this phase of the barrier.
            count_threads[0] -= 1
            # Block Logic: Checks if this is the last thread to reach the barrier.
            # If so, it releases all waiting threads and resets the counter.
            if count_threads[0] == 0:
                i = 0
                # Block Logic: Releases all `num_threads` that are waiting on the semaphore.
                while i < self.num_threads:
                    threads_sem.release()
                    i += 1
                # Inline: Resets the thread count for this phase, allowing reuse of the barrier.
                count_threads[0] = self.num_threads
        # Block Logic: Acquires the semaphore, blocking the current thread until it's released by the last arriving thread.
        threads_sem.acquire()


class Device(object):
    """
    @class Device
    @brief Represents a simulated computational device capable of distributed processing.

    Each `Device` manages its own set of worker threads for parallel script execution,
    sensor data, and coordinates with a `Supervisor` and other devices using barriers.
    It supports dynamic script assignment and data manipulation.

    @attribute max_threads (int): The maximum number of worker threads this device can utilize.
    @attribute device_id (int): A unique identifier for the device.
    @attribute sensor_data (dict): A dictionary holding sensor data, keyed by location.
    @attribute supervisor (Supervisor): A reference to the central supervisor managing all devices.
    @attribute scripts (list): A list of (script, location) tuples to be executed.
    @attribute notification (Event): Event to signal `DeviceThread` that new scripts are available.
    @attribute timepoint_done (Event): Event to signal completion of a timepoint's script execution.
    @attribute update_locks (dict): Dictionary of locks, keyed by location, to protect `set_data` operations.
    @attribute read_locations (dict): Dictionary of events, keyed by location, to signal data readiness for `get_data`.
    @attribute external_barrier (ReusableBarrier): Barrier for synchronizing all `Device` instances.
    @attribute internal_barrier (ReusableBarrier): Barrier for synchronizing worker threads within this device.
    @attribute workers (list): A list of `Worker` thread instances managed by this device.
    @attribute thread (DeviceThread): The dedicated `DeviceThread` for this device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Sets up the device's unique ID, initial sensor data, supervisor reference,
        and various synchronization primitives, worker threads, and data structures.
        It also starts the device's dedicated `DeviceThread`.

        @param device_id (int): A unique identifier for this device.
        @param sensor_data (dict): Initial sensor data for this device, typically a dictionary mapping locations to data values.
        @param supervisor (Supervisor): The supervisor object responsible for managing this device.
        """
        self.max_threads = 8
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.notification = Event()
        self.timepoint_done = Event()
        self.notification.clear()
        self.timepoint_done.clear()
        self.update_locks = {}
        self.read_locations = {}
        self.external_barrier = None
        self.internal_barrier = ReusableBarrier(self.max_threads)
        self.workers = self.setup_workers()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        This method provides a human-readable string that identifies the device
        by its `device_id`.

        @return (str): A string in the format "Device [device_id]".
        """
        return "Device %d" % self.device_id

    def setup_workers(self):
        """
        @brief Initializes and returns a list of worker threads for this device.

        Creates `max_threads` instances of `Worker` threads, each associated
        with this device. These workers will perform script execution.

        @return (list): A list of `Worker` objects.
        """
        workers = []
        i = 0
        # Block Logic: Creates `max_threads` worker instances.
        while i < self.max_threads:
            workers.append(Worker(self))
            i += 1
        return workers

    def start_workers(self):
        """
        @brief Starts all worker threads associated with this device.
        """
        for i in range(0, self.max_threads):
            self.workers[i].start()

    def stop_workers(self):
        """
        @brief Stops all worker threads by signaling them to terminate and joining them.
        """
        for i in range(0, self.max_threads):
            self.workers[i].join()

    def setup_devices(self, devices):
        """
        @brief Configures the external barrier for inter-device synchronization.

        Device 0 initializes a new `ReusableBarrier` for all devices.
        Other devices wait for device 0 to initialize this barrier and then
        adopt it.

        @param devices (list): A list of all `Device` objects in the simulated system.
        """
        # Block Logic: Device 0 initializes the global external barrier.
        if self.device_id == 0:
            self.external_barrier = ReusableBarrier(len(devices))
        else:
            # Block Logic: Other devices wait for device 0 to set up the external barrier.
            for device in devices:
                if device.device_id == 0:
                    # Precondition: Device 0's external barrier must be initialized.
                    # Invariant: Loop until external_barrier is not None.
                    while device.external_barrier is None:
                        pass
                    self.external_barrier = device.external_barrier
                    break

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        This method adds a new (script, location) tuple to the device's
        `scripts` queue and sets the `notification` event to signal
        the `DeviceThread` that there are new scripts to process.
        It also ensures that appropriate locks and events are set up for the location.
        If `script` is None, it sets `timepoint_done`, signaling the end of scripts for a timepoint.

        @param script (object): The script object to be executed.
        @param location (int): The sensor data location that the script pertains to.
        """
        # Block Logic: Signals the DeviceThread that there's an update (either new script or end of timepoint).
        self.notification.set()
        # Block Logic: If a valid script is provided, it's added to the queue, and necessary locks/events are initialized.
        if script is not None:
            # Block Logic: Initializes a new lock and event for the location if they don't exist.
            if location not in self.update_locks:
                self.update_locks[location] = Lock()
                self.read_locations[location] = Event()
                self.read_locations[location].set() # Initially set, meaning data is ready to be read.
            self.scripts.append((script, location))
        else:
            # Block Logic: If no script, it signals that the timepoint's script assignment is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        This method provides access to the sensor data stored on the device
        for a given `location`. It ensures data consistency by waiting for
        any ongoing `set_data` operations to complete for that location.

        @param location (int): The specific location for which to retrieve data.
        @return (any): The data at the specified location, or `None` if the location is not found.
        """
        # Block Logic: Checks if the location exists in sensor data.
        if location not in self.sensor_data:
            return None
        else:
            # Block Logic: Initializes update lock and read event if not already present for this location.
            if location not in self.read_locations:
                self.update_locks[location] = Lock()
                self.read_locations[location] = Event()
                self.read_locations[location].set()
            # Block Logic: Waits for the `read_locations` event to be set, ensuring data is not being written.
            self.read_locations[location].wait()
            return self.sensor_data[location]

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.

        This method updates the `sensor_data` at a given `location` with the
        new `data`. It uses a lock to ensure exclusive access during the write
        operation and signals data unavailability for reads while updating.

        @param location (int): The specific location for which to set data.
        @param data (any): The new data value to be stored at the location.
        """
        # Block Logic: Proceeds only if the location exists in the sensor data.
        if location in self.sensor_data:
            # Block Logic: Acquires the update lock to ensure exclusive write access.
            self.update_locks[location].acquire()
            # Inline: Clears the read event to prevent concurrent reads during data update.
            self.read_locations[location].clear()
            # Inline: Updates the sensor data at the specified location.
            self.sensor_data[location] = data
            # Inline: Sets the read event, signaling that new data is available for reads.
            self.read_locations[location].set()
            # Inline: Releases the update lock.
            self.update_locks[location].release()

    def shutdown(self):
        """
        @brief Shuts down the device's main thread and all associated worker threads.

        This method ensures a clean termination by first stopping all worker
        threads and then joining the device's main thread.
        """
        # Inline: Initiates the shutdown process for all worker threads.
        self.stop_workers()
        # Inline: Waits for the device's main thread to complete its execution.
        self.thread.join()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief Manages the asynchronous execution of scripts on a `Device` using a pool of worker threads.

    This class extends `threading.Thread` to provide a dedicated control thread
    for each `Device` instance. It orchestrates the assignment of scripts to
    available `Worker` threads, handles synchronization at timepoint boundaries,
    and manages the overall simulation flow for the device.

    @attribute device (Device): A reference to the `Device` object this thread is managing.
    """

    def __init__(self, device):
        """
        @brief Initializes a new `DeviceThread` instance.

        Sets up the thread with a descriptive name and stores a reference
        to the `Device` object it will manage.

        @param device (Device): The `Device` instance that this thread will operate on.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def find_free_worker(self):
        """
        @brief Identifies and returns the index of an available worker thread.

        Iterates through the device's worker pool to find a `Worker` thread
        that is currently marked as free (`is_free = True`).

        @return (int): The index of a free worker, or -1 if no worker is currently free.
        """
        # Block Logic: Iterates through the pool of worker threads.
        for i in range(0, self.device.max_threads):
            # Block Logic: Checks if the current worker is free.
            if self.device.workers[i].is_free:
                return i
        # Inline: Returns -1 if no free worker is found after checking all workers.
        return -1

    def run(self):
        """
        @brief The main execution loop for the device's control thread.

        This method continuously executes in a loop, simulating the device's
        operation across multiple timepoints. It performs the following key functions:
        - Starts all worker threads.
        - Continuously retrieves neighbor information from the supervisor.
        - Manages the distribution of scripts to available worker threads.
        - Synchronizes with other devices at timepoint ends using an external barrier.
        - Handles termination of the simulation.
        """
        # Block Logic: Initiates all worker threads for this device.
        self.device.start_workers()

        # Block Logic: Main simulation loop for the device's control thread.
        # Invariant: Each iteration manages script assignment and synchronization for a timepoint.
        while True:
            # Block Logic: Retrieves the current list of neighboring devices from the supervisor.
            # This list is dynamic and may change based on simulation state.
            neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Checks if the simulation should terminate.
            # Precondition: If `neighbours` is None, it indicates the end of the simulation.
            if neighbours is None:
                # Block Logic: Signals all worker threads to terminate.
                for i in range(0, self.device.max_threads):
                    self.device.workers[i].update(None, None, None, "end")
                break # Inline: Exits the main simulation loop.

            # Block Logic: Waits for a notification indicating new scripts are available or a timepoint is done.
            # Precondition: This ensures the device thread does not busy-wait when no tasks are assigned.
            if len(self.device.scripts) == 0:
                self.device.notification.wait()

            # Block Logic: Manages the assignment of scripts to available worker threads.
            curr_scr = 0
            # Invariant: Continues as long as there are scripts to process or the timepoint is not explicitly marked done.
            while (curr_scr < len(self.device.scripts)) or \
                  (self.device.timepoint_done.is_set() is False):
                # Inline: Attempts to find a free worker thread.
                worker_idx = self.find_free_worker()
                # Block Logic: If a free worker is found and there are scripts pending, assign a script.
                if (worker_idx >= 0) and (curr_scr < len(self.device.scripts)):
                    # Inline: Deconstructs the script and location from the current script entry.
                    (script, location) = self.device.scripts[curr_scr]
                    # Inline: Updates the found worker with the script, location, neighbors, and "run" mode.
                    self.device.workers[worker_idx].update(location, script, neighbours, "run")
                    curr_scr += 1 # Inline: Moves to the next script.
                else:
                    continue # Inline: Continues to check for free workers or pending scripts.

            # Block Logic: Signals all worker threads that the timepoint's script processing is complete.
            # This prepares workers for internal barrier synchronization.
            for i in range(0, self.device.max_threads):
                self.device.workers[i].update(None, None, None, "timepoint_end")
            # Inline: Clears the timepoint_done and notification events for the next timepoint.
            self.device.timepoint_done.clear()
            self.device.notification.clear()

            # Block Logic: Synchronizes all devices across the distributed system.
            # All devices wait here until every device has finished its timepoint processing.
            self.device.external_barrier.wait()


class Worker(Thread):
    """
    @class Worker
    @brief A dedicated thread for executing scripts on a `Device`.

    Each `Worker` operates under the direction of its parent `DeviceThread`,
    performing script execution, collecting data from neighbors, and
    updating sensor data for a specific location. It uses internal events
    for coordination with the `DeviceThread`.

    @attribute device (Device): A reference to the parent `Device` object.
    @attribute init_start (Event): Event to signal that the worker is ready to receive new tasks.
    @attribute exec_start (Event): Event to signal that the worker has received a task and should start execution.
    @attribute location (int): The sensor data location this worker is currently processing.
    @attribute script (object): The script object to be executed.
    @attribute neighbours (list): A list of neighboring `Device` objects for data exchange.
    @attribute is_free (bool): Flag indicating whether the worker is available for new tasks.
    @attribute mode (str): The current operational mode of the worker (e.g., "run", "end", "timepoint_end").
    """

    def __init__(self, device):
        """
        @brief Initializes a new `Worker` thread instance.

        Sets up the worker with a reference to its parent device and initializes
        its internal state, including events for task management.

        @param device (Device): The `Device` instance that this worker belongs to.
        """
        Thread.__init__(self)
        self.device = device
        self.init_start = Event()
        self.exec_start = Event()
        self.location = None
        self.script = None
        self.neighbours = None
        self.is_free = True
        self.mode = ""
        self.exec_start.clear() # Inline: Initially clear, meaning execution is not started.
        self.init_start.set()   # Inline: Initially set, meaning worker is ready to receive init.

    def update(self, location, script, neighbours, mode):
        """
        @brief Updates the worker's task parameters and triggers execution.

        This method is called by the `DeviceThread` to assign a new script
        and related context to the worker, changing its mode of operation.

        @param location (int): The sensor data location for the script.
        @param script (object): The script to be executed.
        @param neighbours (list): List of neighboring devices for data access.
        @param mode (str): The operational mode for the worker ("run", "end", "timepoint_end").
        """
        # Block Logic: Waits until the worker is initialized and ready to receive an update.
        self.init_start.wait()
        # Inline: Clears the `init_start` event, indicating the worker is now being updated.
        self.init_start.clear()
        # Block Logic: Assigns the new task parameters to the worker.
        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.mode = mode
        self.is_free = False    # Inline: Marks the worker as busy.
        self.exec_start.set()   # Inline: Sets `exec_start` to trigger the worker's `run` method.

    def run(self):
        """
        @brief The main execution loop for the worker thread.

        This method continuously waits for tasks from the `DeviceThread`.
        Depending on the assigned `mode`, it either:
        - Executes a script, collects data from neighbors, and updates sensor data.
        - Synchronizes with other workers via the internal barrier.
        - Terminates itself.
        """
        # Block Logic: Main loop for the worker thread, continuously processing tasks.
        while True:
            # Block Logic: Waits for a signal to start execution (i.e., a task has been assigned).
            self.exec_start.wait()
            # Inline: Clears the `exec_start` event to indicate that execution has started.
            self.exec_start.clear()
            # Block Logic: Handles the "end" mode, which terminates the worker thread.
            if self.mode == "end":
                break # Inline: Exits the worker's main loop.
            # Block Logic: Handles the "timepoint_end" mode, synchronizing with other workers.
            elif self.mode == "timepoint_end":
                # Block Logic: Waits at the internal barrier, synchronizing all worker threads within this device.
                self.device.internal_barrier.wait()
                self.is_free = True     # Inline: Marks the worker as free after synchronization.
                self.init_start.set()   # Inline: Signals that the worker is ready for a new task.
            else: # Block Logic: Handles the "run" mode, where a script is executed.
                script_data = []

                # Block Logic: Collects sensor data from neighboring devices for the specified location.
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        script_data.append(data)

                # Block Logic: Collects sensor data from its own device for the specified location.
                data = self.device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
                # Block Logic: If any data was collected, executes the script and updates data.
                if script_data != []:
                    # Inline: Executes the assigned script with the collected data.
                    result = self.script.run(script_data)
                    # Block Logic: Updates sensor data on neighboring devices with the script's result.
                    for device in self.neighbours:
                        device.set_data(self.location, result)

                    # Block Logic: Updates sensor data on its own device with the script's result.
                    self.device.set_data(self.location, result)
                self.is_free = True     # Inline: Marks the worker as free after script execution.
                self.init_start.set()   # Inline: Signals that the worker is ready for a new task.
