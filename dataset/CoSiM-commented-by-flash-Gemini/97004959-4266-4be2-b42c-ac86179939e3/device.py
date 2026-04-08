




"""
@97004959-4266-4be2-4b2c-ac86179939e3/device.py
@brief Defines classes for device simulation in a distributed system, including reusable barrier synchronization and script execution management.
Functional Utility: This module orchestrates device behavior in a simulated distributed environment, enabling synchronized data processing and script execution across multiple interconnected devices.
Domain: Distributed Systems, Concurrency, Simulation.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue

class ReusableBarrier(object):
    """
    @class ReusableBarrier
    @brief Implements a reusable barrier synchronization primitive for a fixed number of threads.

    Functional Utility: This barrier allows a group of `num_threads` to halt execution at a specific
    point (`wait` method) until all threads in the group have arrived. Once all
    threads have reached the barrier, they are all released simultaneously.
    The barrier can then be reset and reused for subsequent synchronization points.

    The implementation uses a two-phase approach to ensure reusability without
    deadlock. Each phase involves:
    - A `count_lock` to atomically decrement a thread counter.
    - A `Semaphore` to block arriving threads and release them when the counter reaches zero.

    Domain: Concurrency, Distributed Systems.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes a new instance of the ReusableBarrier.
        @param num_threads (int): The total number of threads expected to participate in the barrier.
        Functional Utility: Sets up the initial state of the barrier, including thread counters,
                            a lock for atomic operations, and semaphores for phase-based synchronization.
        Pre-conditions: `num_threads` must be a positive integer representing the number of threads
                        that will use this barrier.
        Attributes:
            num_threads (int): The total number of threads expected to participate in the barrier.
            count_threads1 (list): Internal counter for the first phase of the barrier.
                                   Initialized to `num_threads`.
            count_threads2 (list): Internal counter for the second phase of the barrier.
                                   Initialized to `num_threads`.
            count_lock (Lock): A lock to protect atomic access to `count_threads1` and `count_threads2`.
            threads_sem1 (Semaphore): Semaphore for the first phase of thread synchronization.
                                      Initialized to 0, blocking threads until all arrive.
            threads_sem2 (Semaphore): Semaphore for the second phase of thread synchronization.
                                      Initialized to 0, blocking threads until all arrive.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks until all `num_threads` threads have arrived at the barrier for both phases.
        Functional Utility: Orchestrates the two-phase synchronization, ensuring all participating threads
                            are synchronized before proceeding.
        Pre-conditions: This method should be called by `num_threads` distinct threads.
        Post-conditions: All threads that called `wait` are released simultaneously after both phases complete.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the barrier synchronization.
        @param count_threads (list): A list containing the current thread counter for this phase.
                                     (Using a list to allow modification within the `with` block).
        @param threads_sem (Semaphore): The semaphore associated with this phase for blocking and releasing threads.
        Functional Utility: Manages the arrival and release of threads for a single phase of the barrier.
                            Threads acquire the semaphore to wait, and the last arriving thread releases
                            all waiting threads.
        Pre-conditions: `count_threads` is a list containing a single integer representing the number
                        of threads yet to arrive in this phase, initialized to `num_threads`.
                        `threads_sem` is a semaphore initialized to 0.
        Post-conditions: All threads that entered this phase are released, and the `count_threads`
                         is reset for reuse.
        """
        with self.count_lock:
            # Block Logic: Decrements the thread counter for the current phase.
            # Invariant: `count_threads[0]` always represents the number of threads
            # still needing to arrive at this phase of the barrier.
            count_threads[0] -= 1
            
            # Block Logic: Checks if this is the last thread to arrive at the barrier.
            # If so, it releases all waiting threads and resets the counter for reuse.
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                
                # Block Logic: Resets the thread counter for the next use of the barrier.
                # Invariant: `count_threads[0]` is reset to `num_threads` to prepare for the next synchronization cycle.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    @class Device
    @brief Represents a simulated device in a distributed environment, capable of executing scripts
           and interacting with sensor data and other devices.

    Functional Utility: Manages a device's state, including its ID, sensor data, communication
                        with a supervisor, script processing, and synchronization mechanisms
                        like locks and barriers for interaction with other devices.
    Domain: Distributed Systems, Simulation, Sensor Networks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary holding sensor data, typically keyed by location.
        @param supervisor (Supervisor): An object responsible for overseeing device interactions,
                                         e.g., providing neighbor information.
        Functional Utility: Sets up the core identity, data, and communication interfaces for a simulated device.
        Attributes:
            device_id (int): Unique identifier for this device.
            sensor_data (dict): Stores the sensor readings and internal state of the device.
            supervisor (Supervisor): Reference to the central supervisor managing the simulation.
            result_queue (Queue.Queue): A queue to store results of script executions.
            set_lock (Lock): A lock to protect `sensor_data` during write operations (set_data).
            neighbours_lock (Lock): A shared lock, initialized by the first device, to protect
                                    access to neighbor information.
            neighbours_barrier (ReusableBarrier): A shared barrier, initialized by the first device,
                                                  to synchronize device threads.
            script_received (Event): An event flag that signals when new scripts have been assigned.
            scripts (list): A list to hold `(script, location)` tuples for execution.
            timepoint_done (Event): An event flag that signals when a timepoint's processing is complete.
            thread (DeviceThread): The dedicated thread for this device's operations.
        Pre-conditions: `device_id` is an integer, `sensor_data` is a dictionary, `supervisor` is a valid object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.result_queue = Queue.Queue()
        self.set_lock = Lock()
        self.neighbours_lock = None
        self.neighbours_barrier = None

        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        Functional Utility: Provides a human-readable identifier for the device, useful for logging and debugging.
        @returns (str): A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device's shared synchronization primitives (lock and barrier)
               based on its `device_id`.
        @param devices (list): A list of all Device instances in the simulation.
        Functional Utility: Ensures that all devices share the same lock and barrier instances,
                            which are initialized by the device with the lowest ID (assumed to be the first in the list).
                            Starts the device's dedicated thread.
        Pre-conditions: `devices` is a non-empty list of `Device` instances.
        Post-conditions: `self.neighbours_lock` and `self.neighbours_barrier` are initialized
                         and shared across all devices. The `self.thread` is started.
        """
        # Block Logic: The first device in the list (typically with device_id 0)
        # initializes the shared lock and barrier.
        if self.device_id == devices[0].device_id:
            self.neighbours_lock = Lock()
            self.neighbours_barrier = ReusableBarrier(len(devices))
        # Block Logic: Subsequent devices reference the shared lock and barrier
        # initialized by the first device.
        else:
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier

        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.
        @param script (Script or None): The script object to be executed, or `None` to signal
                                        that no more scripts are coming for this timepoint.
        @param location (any): The data location relevant to the script.
        Functional Utility: Adds a script and its associated location to the device's queue
                            and signals the `DeviceThread` that new scripts are available.
                            If `script` is `None`, it signals the end of scripts for the current timepoint.
        Pre-conditions: `script` is a callable object with a `run` method, or `None`.
        Post-conditions: The `script` is appended to `self.scripts` if not None, and `self.script_received` is set.
                         If `script` is None, `self.timepoint_done` is also set.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Block Logic: If no script is provided, it indicates the end of scripts
            # for the current timepoint, so both signals are set.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location (any): The key representing the data location.
        Functional Utility: Provides read access to the device's internal `sensor_data` dictionary.
        @returns (any or None): The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a specific location.
        @param location (any): The key representing the data location to update.
        @param data (any): The new data value to set.
        Functional Utility: Atomically updates a data point in the device's `sensor_data` dictionary,
                            ensuring thread safety using `self.set_lock`.
        Pre-conditions: `location` exists as a key in `self.sensor_data`.
        Post-conditions: If `location` is valid, `self.sensor_data[location]` is updated with `data`.
        """
        # Block Logic: Acquires a lock to ensure exclusive write access to `sensor_data`.
        self.set_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_lock.release()

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its thread to complete.
        Functional Utility: Ensures a clean termination of the device's operational thread.
        Post-conditions: The `DeviceThread` associated with this device has completed its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief A dedicated thread that manages the operational lifecycle of a Device instance.

    Functional Utility: This thread continuously fetches neighbor information, waits for and
                        distributes scripts to `DeviceWorker`s, orchestrates their execution,
                        and ensures synchronization using a shared barrier.
    Domain: Concurrency, Distributed Systems, Simulation.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The Device instance that this thread is responsible for.
        Functional Utility: Sets up the thread with a reference to its parent device and
                            initializes a list to hold `DeviceWorker` instances.
        Pre-conditions: `device` is a valid `Device` object.
        Attributes:
            device (Device): The associated `Device` instance.
            workers (list): A list to store `DeviceWorker` threads for script execution.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Orchestrates the continuous process of fetching neighbor data,
                            receiving and distributing scripts to workers, managing worker execution,
                            and synchronizing with other device threads.
        Post-conditions: The loop terminates when `get_neighbours()` returns `None`, indicating shutdown.
        """
        while True:
            # Block Logic: Acquires a lock to safely retrieve neighbor information from the supervisor.
            # This ensures that `get_neighbours` is called in a thread-safe manner.
            self.device.neighbours_lock.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.neighbours_lock.release()

            # Block Logic: Checks if the supervisor has signaled a shutdown by returning None for neighbours.
            # Invariant: If `neighbours` is None, the simulation for this device is ending.
            if neighbours is None:
                break

            # Block Logic: Waits until new scripts have been assigned to the device.
            # This is a blocking call that ensures the thread only proceeds when there's work to do.
            self.device.script_received.wait()

            # Block Logic: Initializes a fresh set of worker threads for the current timepoint's scripts.
            # This ensures that each timepoint starts with a clean slate of worker assignments.
            self.workers = []
            # Block Logic: Creates a fixed number of `DeviceWorker` instances.
            # Each worker is responsible for executing a subset of the assigned scripts.
            for i in range(8):
                self.workers.append(DeviceWorker(self.device, i, neighbours))

            # Block Logic: Distributes assigned scripts among the available `DeviceWorker`s.
            # The distribution strategy aims to balance the load by assigning scripts to workers
            # that already handle data for the script's location, or to the worker with the minimum load.
            for (script, location) in self.device.scripts:

                # Block Logic: Attempts to assign the script to an existing worker that
                # is already handling the data for the script's location.
                added = False
                for worker in self.workers:
                    if location in worker.locations:
                        worker.add_script(script, location)
                        added = True

                # Block Logic: If no worker was found with a matching location, assigns the script
                # to the worker with the fewest assigned locations to balance the workload.
                if added == False:
                    minimum = len(self.workers[0].locations)
                    chosen_worker = self.workers[0]
                    for worker in self.workers:
                        if minimum > len(worker.locations):
                            minimum = len(worker.locations)
                            chosen_worker = worker

                    chosen_worker.add_script(script, location)

            # Block Logic: Starts all `DeviceWorker` threads, initiating parallel script execution.
            for worker in self.workers:
                worker.start()

            # Block Logic: Waits for all `DeviceWorker` threads to complete their assigned scripts.
            # This ensures that all computations for the current timepoint are finished before synchronization.
            for worker in self.workers:
                worker.join()

            # Block Logic: Synchronizes all device threads using the shared `neighbours_barrier`.
            # This ensures that all devices have completed their processing for the current timepoint
            # before any device proceeds to the next timepoint.
            self.device.neighbours_barrier.wait()
            # Block Logic: Clears the `script_received` event, preparing it to wait for
            # new scripts for the next timepoint.
            self.device.script_received.clear()


class DeviceWorker(Thread):
    """
    @class DeviceWorker
    @brief A worker thread responsible for executing a subset of assigned scripts for a Device.

    Functional Utility: Each worker processes scripts pertaining to specific data locations,
                        retrieving data from its own device and neighbors, executing the script,
                        and updating the results back to the relevant devices.
    Domain: Concurrency, Distributed Systems, Simulation, Data Processing.
    """

    def __init__(self, device, worker_id, neighbours):
        """
        @brief Initializes a new DeviceWorker instance.
        @param device (Device): The parent Device instance this worker belongs to.
        @param worker_id (int): A unique identifier for this worker thread within its device.
        @param neighbours (list): A list of neighboring Device instances from which to retrieve data.
        Functional Utility: Sets up the worker with references to its parent device, its ID,
                            and the list of neighbor devices for data interaction.
                            Initializes lists to store scripts and their corresponding locations.
        Pre-conditions: `device` is a valid `Device` object, `worker_id` is an integer,
                        `neighbours` is a list of `Device` objects.
        Attributes:
            device (Device): The associated `Device` instance.
            worker_id (int): Unique identifier for this worker.
            scripts (list): A list to store script objects assigned to this worker.
            locations (list): A list to store data locations associated with each script.
            neighbours (list): A list of `Device` objects representing the neighbors.
        """
        Thread.__init__(self)
        self.device = device
        self.worker_id = worker_id
        self.scripts = []
        self.locations = []
        self.neighbours = neighbours

    def add_script(self, script, location):
        """
        @brief Adds a script and its associated location to the worker's queue for execution.
        @param script (Script): The script object to be executed.
        @param location (any): The data location relevant to the script.
        Functional Utility: Appends the given `script` and `location` to the worker's internal lists,
                            preparing them for processing by `run_scripts`.
        Post-conditions: `script` is added to `self.scripts` and `location` to `self.locations`.
        """
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        """
        @brief Executes all assigned scripts, handling data retrieval and result propagation.
        Functional Utility: Iterates through each script and its location, gathers relevant
                            sensor data from neighboring devices and its own device,
                            executes the script if data is available, and propagates the
                            results back to the neighbors and its own device.
        Pre-conditions: `self.scripts` and `self.locations` are populated with corresponding data.
        Post-conditions: Scripts are executed, and `sensor_data` on involved devices may be updated.
        """
        # Block Logic: Iterates through each assigned script and its associated location.
        for (script, location) in zip(self.scripts, self.locations):

            script_data = []
            
            # Block Logic: Gathers sensor data from neighboring devices for the current location.
            # Invariant: `script_data` will contain data from neighbors that have the specified location.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: Gathers sensor data from the worker's own device for the current location.
            # Invariant: If the worker's device has data for the location, it is added to `script_data`.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script if any relevant data was collected.
            # Functional Utility: Prevents script execution if no data is available, optimizing performance.
            if script_data != []:
                res = script.run(script_data)

                # Block Logic: Propagates the script execution result back to neighboring devices.
                for device in self.neighbours:
                    device.set_data(location, res)
                # Block Logic: Updates the script execution result on the worker's own device.
                self.device.set_data(location, res)

    def run(self):
        """
        @brief The entry point for the DeviceWorker thread.
        Functional Utility: Initiates the execution of all scripts assigned to this worker.
        """
        self.run_scripts()
