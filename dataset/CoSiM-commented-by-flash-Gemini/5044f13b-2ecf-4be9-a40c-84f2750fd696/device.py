"""
@file device.py
@brief This module defines the core components for a distributed device simulation
       framework, including device management, task scheduling, and inter-device
       communication. It orchestrates sensor data processing across multiple
       simulated devices using a supervisor-worker pattern.

Classes:
    - Device: Represents a single simulated device, managing its state, sensor data,
              and interactions within the simulation.
    - DeviceThread: Manages the lifecycle and execution logic for a Device instance.
    - Task: Encapsulates a specific computational or data processing unit of work
            to be performed by a device.
    - TaskScheduler: Distributes and manages Tasks across multiple worker threads,
                     ensuring data consistency through location-specific locking.
    - Worker: A thread responsible for fetching and executing tasks from the
              TaskScheduler's workpool.
"""


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
# Assuming taskscheduler and task are other modules or classes defined elsewhere
# (or locally, as TaskScheduler and Task are defined in this file).


def add_lock_for_location(lock_per_location: list, location: any):
    """
    Functional Utility: Associates a new lock with a specific data location.

    This function adds a tuple containing a data location and a new `threading.Lock`
    instance to a list. This mechanism ensures that concurrent access to sensor
    data at a given `location` can be synchronized.

    Args:
        lock_per_location (list): A list where each element is a tuple `(location, lock_object)`.
                                  This list tracks existing locks for various locations.
        location (any): The identifier for the data location for which a new lock
                        is to be created and associated.
    """
    lock_per_location.append((location, Lock()))

def get_lock_for_location(lock_per_location: list, location: any) -> Lock | None:
    """
    Functional Utility: Retrieves the lock associated with a specific data location.

    This function iterates through a list of location-lock pairs to find and return
    the `threading.Lock` instance corresponding to the given `location`. This is
    crucial for ensuring that only one thread can modify data at a particular location
    at any given time.

    Args:
        lock_per_location (list): A list where each element is a tuple `(location, lock_object)`.
                                  This list tracks existing locks for various locations.
        location (any): The identifier for the data location whose associated lock
                        is to be retrieved.

    Returns:
        threading.Lock or None: The `threading.Lock` object if found for the given
                                location, otherwise None.
    """
    for (loc, lock) in lock_per_location:
        if loc == location:
            return lock
    return None


class Device(object):
    """
    @brief Represents a single simulated device in the distributed system.

    Each `Device` manages its own sensor data, communicates with a supervisor,
    and processes assigned tasks. It uses an `Event` to signal script reception
    and maintains a thread for continuous operation.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings, keyed by location.
        supervisor (object): A reference to the central supervisor managing all devices.
        script_received (Event): A threading.Event to signal when new scripts are assigned.
        scripts (list): A list of tuples, each containing a script and its target location.
        barrier (ReusableBarrierSem): A synchronization barrier for coordinating with other devices.
        taskscheduler (TaskScheduler): A scheduler responsible for dispatching tasks.
        timepoint_done (Event): A threading.Event to signal completion of tasks for a given timepoint.
        thread (DeviceThread): The dedicated thread that runs the device's operational logic.
    """

    def __init__(self, device_id: int, sensor_data: dict, supervisor: object):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): Unique identifier for this device.
            sensor_data (dict): Initial sensor readings for various locations.
            supervisor (object): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None
        self.taskscheduler = None
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self) -> str:
        """
        Returns a string representation of the Device.

        Returns:
            str: A descriptive string including the device ID.
        """
        return "Device %d" % self.device_id

    def share_barrier(self, barrier: ReusableBarrierSem):
        """
        Functional Utility: Shares a synchronization barrier with this device.

        This allows the device to coordinate its execution with other devices
        at specific timepoints in the simulation.

        Args:
            barrier (ReusableBarrierSem): The barrier object to be shared.
        """
        self.barrier = barrier

    def share_taskscheduler(self, taskscheduler: 'TaskScheduler'):
        """
        Functional Utility: Shares a TaskScheduler with this device.

        The TaskScheduler is responsible for receiving tasks generated by this
        device and distributing them to worker threads for execution.

        Args:
            taskscheduler (TaskScheduler): The TaskScheduler object to be shared.
        """
        self.taskscheduler = taskscheduler

    def setup_devices(self, devices: list):
        """
        Functional Utility: Initializes shared resources (barrier and task scheduler)
                            across all participating devices.

        This method is typically called by a designated "master" device (e.g., device_id 0)
        to set up the common infrastructure for inter-device coordination and task distribution.
        It identifies all unique sensor data locations across all devices and creates a
        `threading.Lock` for each to ensure safe concurrent access.

        Pre-condition: This method should only be called by a single, coordinating device.
        Invariant: After execution, all devices will share the same `ReusableBarrierSem`
                   and `TaskScheduler` instance, with location-specific locks initialized.

        Args:
            devices (list): A list of all `Device` objects participating in the simulation.
        """
        if self.device_id == 0: # Block Logic: Ensures only the master device performs setup.
            lock_per_location = [] # Stores (location, Lock()) tuples.

            # Block Logic: Iterates through all devices and their sensor data
            #              to identify unique locations and create a lock for each.
            # Invariant: After each inner loop iteration, `lock_per_location` contains
            #            locks for all locations processed so far for the current device.
            for device in devices:
                for location in device.sensor_data:
                    lock = get_lock_for_location(lock_per_location, location)
                    if lock is None: # Condition Check: Only add a lock if one doesn't already exist for the location.
                        add_lock_for_location(lock_per_location, location)

            # Functional Utility: Initializes the shared barrier and task scheduler.
            # The barrier is sized for all devices.
            self.barrier = ReusableBarrierSem(len(devices))
            self.taskscheduler = TaskScheduler(lock_per_location)

            # Block Logic: Distributes the initialized barrier and task scheduler
            #              to all other devices in the system.
            # Invariant: After each iteration, the current 'device' has been configured
            #            with the shared barrier and taskscheduler.
            for device in devices:
                device.share_taskscheduler(self.taskscheduler)
                device.share_barrier(self.barrier)

    def assign_script(self, script: object, location: any):
        """
        Functional Utility: Assigns a new script to be executed on this device
                            at a specific location, or signals completion.

        If a script is provided, it's added to the device's queue, and an event
        is set to notify the device's thread. If `None` is provided for the script,
        it signals that the current timepoint's tasks are done, allowing the device
        to proceed with synchronization.

        Args:
            script (object): The script object to be executed, or None to signal
                             completion of the current timepoint's assignments.
            location (any): The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signals the device thread that a script has been received.
        else:
            self.timepoint_done.set() # Signals that this device has no more scripts for the current timepoint.

    def get_data(self, location: any):
        """
        Functional Utility: Retrieves sensor data for a specified location.

        Args:
            location (any): The identifier for the data location.

        Returns:
            any: The sensor data if the location exists, otherwise None.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location: any, data: any):
        """
        Functional Utility: Sets or updates sensor data for a specified location.

        Args:
            location (any): The identifier for the data location.
            data (any): The new sensor data to be stored.
        """
        self.sensor_data[location] = data

    def shutdown(self):
        """
        Functional Utility: Shuts down the device by joining its operational thread.
        This ensures all ongoing operations are completed before termination.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the continuous operation and task processing for a `Device`.

    This thread continuously monitors for new scripts, processes them by creating
    tasks for the `TaskScheduler`, and synchronizes with other devices using a barrier.
    It handles the main simulation loop for a specific device.

    Attributes:
        device (Device): The Device instance that this thread is managing.
    """

    def __init__(self, device: 'Device'):
        """
        Initializes a new DeviceThread.

        Args:
            device (Device): The Device instance this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Block Logic: The main execution loop for the device thread.

        It waits for shared resources to be initialized, then enters a continuous
        loop to fetch neighbor information, process assigned scripts into tasks,
        and synchronize with other devices. The loop breaks when the supervisor
        signals the end of the simulation.

        Pre-condition: The associated `Device` instance is properly initialized.
        Invariant: The thread continues to process tasks and synchronize until
                   the simulation is concluded by the supervisor.
        """
        # Block Logic: Waits until the shared barrier is initialized by the master device.
        # Pre-condition: `self.device.barrier` is initially None.
        while self.device.barrier is None:
            pass

        # Block Logic: Continuous loop for simulation timepoints.
        # Invariant: Each iteration represents a new simulation timepoint.
        while True:
            # Functional Utility: Retrieves information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Condition Check: If no neighbors are returned, it signifies the end of the simulation.
            if neighbours is None:
                # Block Logic: If this is the master device, it signals the TaskScheduler to finish
                #              and waits for all worker threads to complete their current tasks.
                if self.device.device_id == 0:
                    self.device.taskscheduler.finish = True
                    self.device.taskscheduler.wait_workers()
                break # Exit the main simulation loop.

            # Block Logic: Waits until all scripts for the current timepoint have been assigned and processed.
            # Pre-condition: `self.device.timepoint_done` is cleared at the start of each timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Resets the event for the next timepoint.

            # Block Logic: Processes each assigned script by creating a `Task` and adding it to the scheduler.
            # Invariant: After each iteration, a script from `self.device.scripts` has been converted
            #            into a `Task` and submitted, clearing `self.device.scripts` after submission.
            # Note: `self.device.scripts` should ideally be cleared after submission or managed as a queue.
            # The current implementation implies that scripts are removed or processed only once.
            for (script, location) in self.device.scripts:
                new_task = Task(self.device, script, location, neighbours)
                self.device.taskscheduler.add_task(new_task)

            # Functional Utility: Synchronizes with other devices, ensuring all devices
            #                     have completed their task assignments for the timepoint.
            self.device.barrier.wait()


class Task(object):
    """
    @brief Represents a unit of work to be executed by a `Worker` thread.

    A `Task` encapsulates a specific script, the data location it operates on,
    and information about neighboring devices relevant for data exchange.

    Attributes:
        device (Device): The Device instance that owns this task.
        script (object): The computational logic to be executed.
        location (any): The data location this task is focused on.
        neighbours (list): A list of neighboring Device objects for data access.
    """

    def __init__(self, device: 'Device', script: object, location: any, neighbours: list):
        """
        Initializes a new Task instance.

        Args:
            device (Device): The Device instance creating this task.
            script (object): The script containing the execution logic.
            location (any): The data location pertinent to this task.
            neighbours (list): List of neighbor devices to interact with.
        """
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def execute(self):
        """
        Block Logic: Executes the encapsulated script using data from the
                     owning device and its neighbors, then updates their data.

        This method gathers relevant sensor data from all involved devices
        (the task's device and its neighbors) for a specific location, runs
        the script with this aggregated data, and then propagates the result
        back to the same devices.

        Pre-condition: `self.device`, `self.script`, `self.location`, and
                       `self.neighbours` are all properly initialized.
        Invariant: After execution, the `sensor_data` at `self.location`
                   for `self.device` and all `self.neighbours` will be updated
                   with the `result` of `self.script.run()`.
        """
        script_data = [] # Accumulates sensor data from relevant devices.

        # Block Logic: Gathers sensor data from neighboring devices at the specified location.
        # Invariant: After each iteration, `script_data` includes data from the current `device`
        #            in `self.neighbours` if available at `self.location`.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Functional Utility: Gathers sensor data from the owning device at the specified location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Condition Check: Ensures there is data to process before running the script.
        if script_data != []:
            # Functional Utility: Executes the script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Propagates the script's result back to all neighboring devices.
            # Invariant: After each iteration, the `sensor_data` for the current `device`
            #            in `self.neighbours` at `self.location` is updated with `result`.
            for device in self.neighbours:
                device.set_data(self.location, result)

            # Functional Utility: Propagates the script's result back to the owning device.
            self.device.set_data(self.location, result)


# The following classes are part of the original file, but due to the prompt structure
# and potential truncation, they might appear as if they were from a separate import.
# They are included here for completeness of the comment generation.
# from threading import Thread, Lock # Already imported at the top.

class TaskScheduler(object):
    """
    @brief Manages the distribution and execution of `Task` objects across a pool of worker threads.

    The `TaskScheduler` maintains a workpool of pending tasks and a set of worker
    threads. It ensures that tasks operating on the same data `location` are
    executed serially by acquiring location-specific locks.

    Attributes:
        nr_threads (int): The number of worker threads to spawn.
        lock_per_location (list): A list of `(location, threading.Lock)` tuples
                                  for ensuring exclusive access to data locations.
        workpool (list): A list of `Task` objects awaiting execution.
        workpool_lock (Lock): A lock to protect access to the `workpool`.
        workers_list (list): A list of `Worker` thread instances.
        finish (bool): A flag to signal worker threads to terminate.
    """

    def __init__(self, lock_per_location: list):
        """
        Initializes a new TaskScheduler instance.

        Args:
            lock_per_location (list): A list of `(location, threading.Lock)` tuples.
        """
        self.nr_threads = 16
        self.lock_per_location = lock_per_location
        self.workpool = []
        self.workpool_lock = Lock()
        self.workers_list = []
        self.finish = False # Control flag for worker threads.

        self.start_workers()

    def add_task(self, new_task: 'Task'):
        """
        Functional Utility: Adds a new task to the workpool for asynchronous execution.

        Args:
            new_task (Task): The `Task` object to be added to the queue.
        """
        # Block Logic: Ensures thread-safe addition of tasks to the workpool.
        with self.workpool_lock:
            self.workpool.append(new_task)

    def get_task(self) -> 'Task' | None:
        """
        Functional Utility: Retrieves a task from the workpool for execution.

        Ensures thread-safe retrieval and removes the task from the pool.

        Returns:
            Task or None: A `Task` object if the workpool is not empty, otherwise None.
        """
        self.workpool_lock.acquire() # Pre-condition: Acquire lock before accessing shared workpool.
        if self.workpool != []:
            ret = self.workpool.pop() # Invariant: The workpool is modified by removing one task.
        else:
            ret = None
        self.workpool_lock.release() # Post-condition: Release lock after modifying or checking workpool.
        return ret

    def start_workers(self):
        """
        Functional Utility: Creates and starts a pool of `Worker` threads.

        Initializes `self.nr_threads` worker threads and adds them to `self.workers_list`,
        then starts each worker thread.
        """
        tid = 0
        # Block Logic: Loop to create and store Worker instances.
        # Invariant: Each iteration creates one Worker and increments `tid`.
        while tid < self.nr_threads:
            thread = Worker(self)
            self.workers_list.append(thread)
            tid += 1

        # Block Logic: Loop to start all created Worker threads.
        # Invariant: After each iteration, one Worker thread has been started.
        for worker in self.workers_list:
            worker.start()

    def wait_workers(self):
        """
        Functional Utility: Waits for all worker threads to complete their execution.

        This method is called during shutdown to ensure all pending tasks are
        processed and threads are properly terminated.
        """
        # Block Logic: Joins each worker thread, blocking until it finishes.
        # Invariant: After each iteration, one worker thread has completed its execution.
        for worker in self.workers_list:
            worker.join()

    def get_lock_per_location(self, location: any) -> Lock | None:
        """
        Functional Utility: Retrieves the `threading.Lock` object associated
                            with a specific data `location`.

        Args:
            location (any): The identifier for the data location.

        Returns:
            threading.Lock or None: The lock object if found, otherwise None.
        """
        for (loc, lock) in self.lock_per_location:
            if loc == location:
                return lock
        return None


class Worker(Thread):
    """
    @brief A worker thread responsible for continuously fetching and executing tasks
           from the `TaskScheduler`.

    Each `Worker` thread operates within the `TaskScheduler`'s context, retrieving
    tasks, acquiring necessary locks for data consistency, and executing the task's
    logic.

    Attributes:
        taskscheduler (TaskScheduler): A reference to the TaskScheduler that
                                       manages this worker.
    """

    def __init__(self, taskscheduler: 'TaskScheduler'):
        """
        Initializes a new Worker instance.

        Args:
            taskscheduler (TaskScheduler): The TaskScheduler this worker belongs to.
        """
        Thread.__init__(self)
        self.taskscheduler = taskscheduler

    def run(self):
        """
        Block Logic: The main execution loop for the worker thread.

        The worker continuously attempts to retrieve and execute tasks from the
        `TaskScheduler`'s workpool. It respects the `TaskScheduler.finish` flag
        to gracefully terminate. Before executing a task, it acquires the
        appropriate location-specific lock to prevent race conditions.

        Pre-condition: The `TaskScheduler` is initialized and has worker threads started.
        Invariant: The worker continues to fetch and execute tasks, respecting locks,
                   until the `TaskScheduler` signals termination.
        """
        # Block Logic: Main loop for worker activity. Breaks when scheduler signals finish.
        while True:
            if self.taskscheduler.finish == True: # Condition Check: Checks if the scheduler has signaled for termination.
                break # Exit the worker thread's main loop.

            # Block Logic: Inner loop to process all available tasks in the workpool before re-checking finish flag.
            # Invariant: Continues to fetch and execute tasks until the workpool is empty.
            while True:
                task = self.taskscheduler.get_task()
                if task is None: # Condition Check: If no task is available, break the inner loop and re-check `finish`.
                    break
                # Block Logic: Acquires a location-specific lock before executing the task.
                # This ensures that only one worker can modify data at a specific location at a time.
                # Pre-condition: A lock exists for `task.location`.
                # Invariant: The task is executed only after successfully acquiring the lock,
                #            and the lock is released automatically upon exiting the `with` block.
                with self.taskscheduler.get_lock_per_location(task.location):
                    task.execute()

