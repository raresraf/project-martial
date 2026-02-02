


"""
@file device.py
@brief Implements a simulated device for distributed processing, managing sensor data, scripts, and parallel execution.

This module defines classes for a distributed system:
- `Device`: Represents an individual processing unit with local data and scripts.
- `WorkPool`: Manages a pool of worker threads for parallel task execution on a device.
- `Worker`: A thread that fetches and executes scripts on sensor data.
- `DeviceThread`: Orchestrates the overall lifecycle of a device, including synchronization with a supervisor and script execution.

Functional Utility: Provides a framework for simulating edge devices or nodes in a distributed computing environment,
handling asynchronous task execution and data synchronization.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


    """
    @class Device
    @brief Represents a simulated device in a distributed system.

    Functional Utility: Manages its unique identifier, local sensor data, and a reference
    to a supervisor entity. It orchestrates script reception and execution,
    and synchronizes its state with other devices through events and barriers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the sensor data managed by this device.
        @param supervisor: A reference to the supervisor object that manages this device.
        Functional Utility: Sets up the device's identity, its data, and establishes
        communication channels (Events) for coordinating with other components.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a script has been received and assigned to the device.
        self.script_received = Event()
        # List to store assigned scripts and their locations.
        self.scripts = []
        # Event to signal that processing for a timepoint is complete.
        self.timepoint_done = Event()
        # Thread that runs the device's main logic.
        self.thread = None
        # WorkPool instance for managing worker threads and tasks.
        self.work_pool = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        Functional Utility: Provides a human-readable identifier for the device.
        @return: A string in the format "Device %d" % self.device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the work pools and threads for a collection of devices.
        @param devices: A list of Device objects to be set up.
        Functional Utility: This method is typically called by the supervisor
        or a central coordinator. It initializes a shared barrier for synchronization
        across all devices and creates individual WorkPools and DeviceThreads for each.
        Pre-condition: Assumes `self.device_id == 0` for the coordinating device.
        """
        # Block Logic: Only the device with ID 0 is responsible for setting up all devices.
        # Invariant: A single barrier and a list of lock locations are created once for all devices.
        if self.device_id == 0:
            # Reusable barrier for synchronizing all device threads.
            barrier = ReusableBarrierCond(len(devices))
            # List of Locks, one for each unique data location across all devices, to ensure exclusive access.
            lock_locations = []

            # Block Logic: Initialize a lock for each sensor data location across all devices.
            # Invariant: Each unique location in `sensor_data` across all devices gets a dedicated lock.
            for device in devices:
                for _ in xrange(len(device.sensor_data)):
                    lock_locations.append(Lock())

            # Block Logic: For each device, create a WorkPool and a dedicated DeviceThread.
            # Invariant: Each device gets its own work management system and runs in a separate thread.
            for device in devices:
                # Event to signal when all tasks for a timepoint are finished for a device.
                tasks_finish = Event()
                device.work_pool = WorkPool(tasks_finish, lock_locations)

                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed on a specific data location.
        @param script: The script object to be executed.
        @param location: The data location (key in sensor_data) where the script should operate.
        Functional Utility: Adds a script and its target location to the device's queue
        for later processing by its worker pool. Signals `script_received` event if a script is assigned.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key corresponding to the desired sensor data.
        @return: The sensor data at the specified location, or None if the location is not found.
        Functional Utility: Provides read access to the device's local sensor data store.
        """
        return self.sensor_data[location] if location in self.sensor_data \
                else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.
        @param location: The key corresponding to the sensor data to be updated.
        @param data: The new data value to set.
        Functional Utility: Provides write access to the device's local sensor data store.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's associated thread.
        Functional Utility: Ensures proper termination and cleanup of the DeviceThread.
        """
        self.thread.join()

class WorkPool(object):

"""
        self.thread.join()

class WorkPool(object):
    """
    @class WorkPool
    @brief Manages a pool of worker threads for executing tasks on a device.

    Functional Utility: Distributes tasks among a fixed number of worker threads,
    ensures proper synchronization for task retrieval and completion, and
    manages access to shared data locations via locks.
    """

    def __init__(self, tasks_finish, lock_locations):
        """
        @brief Initializes a new WorkPool instance.
        @param tasks_finish: An Event that is set when all assigned tasks are completed.
        @param lock_locations: A list of Lock objects, one for each data location,
                               to manage concurrent access.
        Functional Utility: Sets up the worker threads, initializes task queues,
        and establishes synchronization mechanisms for task management.
        """

        self.workers = []
        self.tasks = []
        self.current_task_index = 0
        self.lock_get_task = Lock()
        # Event to signal workers that there is work available.
        self.work_to_do = Event()
        self.tasks_finish = tasks_finish
        self.lock_locations = lock_locations
        self.max_num_workers = 8

        # Block Logic: Create and start a fixed number of worker threads.
        # Invariant: The work pool maintains a consistent number of active worker threads.
        for i in xrange(self.max_num_workers):
            self.workers.append(Worker(self, self.lock_get_task, \
                    self.work_to_do, self.lock_locations))
            self.workers[i].start()

    def set_tasks(self, tasks):
        """
        @brief Assigns a new set of tasks to the work pool.
        @param tasks: A list of task tuples (script, location, neighbours, self_device).
        Functional Utility: Clears previous tasks, sets new tasks, and signals
        the worker threads that new work is available.
        """
        self.tasks = tasks
        self.current_task_index = 0

        self.work_to_do.set()

    def get_task(self):
        """
        @brief Retrieves the next available task from the queue.
        @return: A task tuple (script, location, neighbours, self_device) or None if no tasks are left.
        Functional Utility: Provides a synchronized mechanism for worker threads to
        obtain tasks. It updates the task index and signals task completion when the queue is empty.
        Pre-condition: This method should be called within a locked section (`lock_get_task`).
        """
        # Block Logic: Checks if there are any remaining tasks to process.
        # Invariant: `current_task_index` is always less than or equal to `len(self.tasks)`.
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index]

            self.current_task_index = self.current_task_index + 1

            # Block Logic: If all tasks are assigned, clear the work signal and set the finish signal.
            # Invariant: `work_to_do` is cleared and `tasks_finish` is set only when the last task is distributed.
            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear()
                self.tasks_finish.set()

            return task
        else:
            return None

    def close(self):
        """
        @brief Shuts down the work pool and its worker threads.
        Functional Utility: Empties the task queue, signals workers to terminate,
        and waits for all worker threads to complete their execution.
        """
        self.tasks = []
        self.current_task_index = len(self.tasks)

        # Functional Utility: Signals workers to wake up and check for termination condition.
        self.work_to_do.set()


        # Block Logic: Joins each worker thread, ensuring they complete their execution gracefully.
        for worker in self.workers:
            worker.join()

class Worker(Thread):
    """
    @class Worker
    @brief A thread that executes scripts on sensor data.

    Functional Utility: Continuously fetches tasks from a shared WorkPool,
    acquires necessary locks for data access, executes the assigned script,
    and updates sensor data on relevant devices. It terminates gracefully
    when no more tasks are available.
    """

    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        """
        @brief Initializes a new Worker thread.
        @param work_pool: The WorkPool instance from which to fetch tasks.
        @param lock_get_task: A Lock to synchronize access to the WorkPool's task queue.
        @param work_to_do: An Event that signals when new tasks are available in the WorkPool.
        @param lock_locations: A list of Locks to manage access to individual data locations.
        Functional Utility: Sets up the worker with references to the shared resources
        needed for task acquisition and data manipulation.
        """

        Thread.__init__(self)
        self.work_pool = work_pool
        self.lock_get_task = lock_get_task
        self.work_to_do = work_to_do
        self.lock_locations = lock_locations

    def run(self):
        """
        @brief The main execution loop for the Worker thread.
        Functional Utility: Continuously attempts to acquire a task, executes it,
        and releases resources. It waits if no tasks are available and terminates
        when the WorkPool signals its closure.
        """
        while True:
            # Block Logic: Acquire lock to safely access the WorkPool's task queue.
            # Pre-condition: `lock_get_task` is available.
            self.lock_get_task.acquire()

            # Block Logic: Wait until the WorkPool signals that there is work to do.
            # Invariant: The worker only proceeds when `work_to_do` event is set.
            self.work_to_do.wait()

            task = self.work_pool.get_task()

            # Block Logic: Release lock after retrieving a task.
            self.lock_get_task.release()

            # Block Logic: If no task is returned (WorkPool is closing), break the loop and terminate.
            if task is None:
                break

            # Functional Utility: Unpack task components for script execution.
            script = task[0]
            location = task[1]
            neighbours = task[2]
            self_device = task[3]

            # Block Logic: Acquire lock for the specific data location to ensure exclusive access during script execution.
            # Pre-condition: `location` corresponds to a valid index in `lock_locations`.
            self.lock_locations[location].acquire()

            script_data = []

            # Block Logic: Collect relevant sensor data from neighboring devices for script input.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            # Functional Utility: Collect sensor data from the current device for script input.
            data = self_device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Execute the script if there is input data and update relevant devices.
            if script_data != []:
                # Functional Utility: Execute the script with the collected data.
                result = script.run(script_data)

                # Block Logic: Update sensor data on neighboring devices with the script's result.
                for device in neighbours:
                    device.set_data(location, result)
                # Functional Utility: Update sensor data on the current device with the script's result.
                self_device.set_data(location, result)

            # Block Logic: Release the lock for the data location after script execution and data updates.
            self.lock_locations[location].release()



class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief Manages the lifecycle and synchronization for a single Device.

    Functional Utility: Orchestrates the device's main loop, ensuring that
    it synchronizes with a global barrier, receives and processes scripts,
    and signals its completion for each timepoint.
    """

    def __init__(self, device, barrier, tasks_finish):
        """
        @brief Initializes a new DeviceThread.
        @param device: The Device object that this thread manages.
        @param barrier: A ReusableBarrierCond for global synchronization across all devices.
        @param tasks_finish: An Event that signals when all tasks for the device's current timepoint are completed.
        Functional Utility: Configures the thread with its associated device and
        the necessary synchronization primitives.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.tasks_finish = tasks_finish

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Coordinates the device's operations through different phases:
        (1) waiting for global synchronization, (2) retrieving neighbor information,
        (3) waiting for scripts to be assigned, and (4) processing those scripts
        using its WorkPool. It terminates when the supervisor signals shutdown.
        """
        while True:
            # Block Logic: Synchronize with all other device threads before proceeding to the next timepoint.
            # Invariant: All device threads must reach this barrier before any can continue.
            self.barrier.wait()

            # Functional Utility: Get information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Check if the supervisor has signaled a shutdown.
            # Pre-condition: `neighbours` will be None if shutdown is initiated.
            if neighbours is None:
                self.device.work_pool.close()
                break

            # Block Logic: Wait for scripts to be assigned by the supervisor for the current timepoint.
            # Invariant: The device will not proceed until `script_received` event is set.
            self.device.script_received.wait()

            # Functional Utility: Prepare tasks for the WorkPool based on assigned scripts.
            tasks = []

            # Block Logic: Populate the task list from the scripts assigned to this device.
            for (script, location) in self.device.scripts:
                tasks.append((script, location, neighbours, self.device))

            # Block Logic: If there are tasks, assign them to the work pool and wait for their completion.
            if tasks != []:
                self.device.work_pool.set_tasks(tasks)

                # Block Logic: Wait until all tasks for the current timepoint are finished by the WorkPool's workers.
                # Invariant: The device thread will block here until `tasks_finish` is set by the WorkPool.
                self.tasks_finish.wait()

                # Functional Utility: Clear the `tasks_finish` event for the next timepoint.
                self.tasks_finish.clear()

            # Functional Utility: Clear the `script_received` event for the next timepoint, awaiting new script assignments.
            self.device.script_received.clear()
