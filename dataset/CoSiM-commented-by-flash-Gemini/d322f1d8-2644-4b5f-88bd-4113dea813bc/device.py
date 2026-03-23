"""
@file device.py
@brief Implements a distributed device simulation framework where each device has its
       own work pool of threads to process assigned scripts, and all devices
       synchronize globally at timepoints using a conditional barrier.
       This module orchestrates data processing across multiple simulated devices
       that can interact with their neighbors, with a centralized management of global locks.

Algorithm:
- **Device Initialization:** Each `Device` holds local sensor data, scripts, and references
  to its supervisor and event objects for internal signaling. Its `DeviceThread` and `WorkPool`
  are initialized during `setup_devices`.
- **Centralized Setup by Device 0:** The device with `device_id == 0` acts as a coordinator.
  It initializes a single `ReusableBarrierCond` for all devices and a shared list of global locks (`lock_locations`),
  one for each potential data location across all devices.
- **Per-Device Work Pools:** Each `Device` then gets its own `WorkPool` instance, initialized with
  a reference to the shared `lock_locations` and a `tasks_finish` event for signaling completion.
- **DeviceThread Lifecycle:**
    1. Each `DeviceThread` starts by waiting at the global `ReusableBarrierCond`, ensuring all devices
       are synchronized at the beginning of a timepoint.
    2. It fetches its `neighbours` from the supervisor. If `None`, it signals its `work_pool` to close and exits.
    3. It waits for the `script_received` event (set by the supervisor or `assign_script` when all scripts
       for a timepoint are ready).
    4. It aggregates its assigned scripts into a list of tasks. Each task includes the script, location,
       neighbors, and a reference to itself.
    5. It submits these tasks to its own `WorkPool` using `set_tasks`.
    6. It then waits for its `tasks_finish` event to be set by its `WorkPool`, indicating all its tasks are done.
    7. Finally, it clears the `tasks_finish` and `script_received` events for the next timepoint.
- **WorkPool & Worker Interaction:**
    1. The `WorkPool` manages a fixed number of `Worker` threads.
    2. When `set_tasks` is called, it populates its internal task list and sets an `Event` (`work_to_do`)
       to wake up waiting `Worker`s.
    3. Each `Worker` continuously acquires a lock (`lock_get_task`), waits for `work_to_do`, and
       then safely retrieves a task from the `WorkPool`'s shared task list.
    4. For each task, a `Worker` acquires the specific global lock from `lock_locations` corresponding
       to the task's data location.
    5. It gathers data from `self_device` and `neighbours`, executes the `script.run()` method,
       and updates data on `self_device` and `neighbours`.
    6. It releases the global data lock.
    7. Once all tasks are processed by a `WorkPool`, it sets the `tasks_finish` event.

Time Complexity:
- The overall time complexity is highly dependent on:
    - The number of devices (N).
    - The number and complexity of scripts (S) per device.
    - The network topology (number of neighbors per device).
    - The number of worker threads in each `WorkPool`.
- Script execution is parallelized by the per-device `WorkPool`s.
- `ReusableBarrierCond.wait()` is a global synchronization point, bottlenecking if any device is slow.

Space Complexity:
- O(N_devices * (D + S)) for storing device-specific sensor data (D) and scripts (S).
- Additional overhead for:
    - Global `lock_locations` list (proportional to total unique data locations across all devices).
    - `ReusableBarrierCond` internal structures.
    - `WorkPool` and its `Worker` threads' internal state per device.
"""

from threading import Event, Thread, Lock # Import necessary threading primitives.
from barrier import ReusableBarrierCond   # Import the custom ReusableBarrierCond class.


class Device(object):
    """
    @class Device
    @brief Represents a single simulated device in the distributed system.
           Manages its local sensor data, assigned scripts, and synchronization mechanisms.
           It interacts with a central supervisor and its own dedicated work pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id: Unique identifier for this device.
        @param sensor_data: A dictionary containing sensor readings/data specific to this device.
        @param supervisor: The central supervisor object responsible for managing all devices.
        """
        self.device_id = device_id     # Unique identifier for the device.
        self.sensor_data = sensor_data # Local sensor data (dict: {'location_id': data_value}).
        self.supervisor = supervisor   # Reference to the supervisor object.

        self.script_received = Event() # Event to signal when new scripts have been assigned.
        self.scripts = []              # List to store assigned scripts (script_obj, location_id).
        self.timepoint_done = Event()  # Event for internal device synchronization (less critical with global barrier).
        
        self.thread = None             # Placeholder for the DeviceThread instance.
        self.work_pool = None          # Placeholder for the WorkPool instance owned by this device.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives and per-device work pools.
               Device 0 initializes global resources (barrier, shared locks);
               all devices then initialize their own `DeviceThread` and `WorkPool`.
        @param devices: A list of all Device instances in the simulation.
        Functional Utility: Coordinates the initialization of global and per-device
                            threading and synchronization components.
        """
        # Block Logic: Device 0 acts as the coordinator to initialize global shared resources.
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices)) # Initialize a reusable conditional barrier for all devices.
            lock_locations = [] # List to hold global locks for all potential data locations.

            # Block Logic: Create a Lock for each unique data location across all devices.
            # Invariant: `lock_locations` will contain one Lock for each `sensor_data` entry,
            #            assuming locations are implicitly mapped to indices. This can be brittle.
            for device in devices:
                for _ in xrange(len(device.sensor_data)): # xrange for Python 2 compatibility.
                    lock_locations.append(Lock())

            # Block Logic: Initialize `DeviceThread` and `WorkPool` for each device.
            # Invariant: Each device gets its dedicated `DeviceThread` and `WorkPool`,
            #            sharing the global `barrier` and `lock_locations`.
            for device in devices:
                tasks_finish = Event() # Event for this device's WorkPool to signal task completion.
                device.work_pool = WorkPool(tasks_finish, lock_locations) # Create a WorkPool for the device.

                # Create and start the DeviceThread, passing shared barrier and its tasks_finish event.
                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific data location.
        @param script: The script object to be executed (expected to have a `run` method).
        @param location: The data location (key) the script operates on.
        Functional Utility: Queues scripts for processing by the device's WorkPool.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
        else:
            # If `script` is None, it acts as a signal that all scripts for the current
            # timepoint have been assigned, allowing the DeviceThread to proceed.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
               Note: This implementation of get_data does not use `self.slock` for protection.
        @param location: The key corresponding to the data location.
        @return The data at the specified location, or None if the location does not exist.
        Functional Utility: Provides access to the device's local sensor data.
        """
        return self.sensor_data[location] if location in self.sensor_data 
                else None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a specific location.
               Note: This implementation of set_data does not use `self.slock` for protection.
        @param location: The key corresponding to the data location.
        @param data: The new data value to set.
        Functional Utility: Modifies the device's local sensor data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's thread.
        Functional Utility: Ensures proper termination and cleanup of the `DeviceThread`.
        """
        self.thread.join() # Waits for the DeviceThread to complete its execution.


class WorkPool(object):
    """
    @class WorkPool
    @brief Manages a pool of worker threads (`Worker`) for a single device to execute tasks.
           It distributes tasks to workers, signals task completion, and manages global data locks.
    """

    def __init__(self, tasks_finish, lock_locations):
        """
        @brief Initializes the WorkPool with worker threads and shared synchronization objects.
        @param tasks_finish: An `Event` that the WorkPool sets when all its current tasks are finished.
        @param lock_locations: A shared list of global `Lock` objects, one for each data location.
        """
        self.workers = []                  # List to hold worker thread instances.
        self.tasks = []                    # List to store tasks assigned to this WorkPool.
        self.current_task_index = 0        # Index to track the next task to be given to a worker.
        self.lock_get_task = Lock()        # Lock to protect `current_task_index` and `tasks` access.
        self.work_to_do = Event()          # Event to signal workers that there is work available.
        self.tasks_finish = tasks_finish   # Event to signal the DeviceThread when all tasks are done.
        self.lock_locations = lock_locations # Shared list of global locks for data locations.
        self.max_num_workers = 8           # Maximum number of worker threads in this pool.

        # Block Logic: Create and start the specified number of worker threads.
        for i in xrange(self.max_num_workers): # xrange for Python 2 compatibility.
            # Each worker is initialized with references to the WorkPool's shared resources.
            self.workers.append(Worker(self, self.lock_get_task, 
                    self.work_to_do, self.lock_locations))
            self.workers[i].start()

    def set_tasks(self, tasks):
        """
        @brief Assigns a new set of tasks to the WorkPool for execution.
        @param tasks: A list of tasks (e.g., tuples containing script, location, neighbors, self_device).
        Functional Utility: Resets the task queue and signals worker threads to start processing.
        """
        self.tasks = tasks                 # Overwrite the current task list.
        self.current_task_index = 0        # Reset task index for the new set of tasks.
        self.work_to_do.set()              # Signal workers that new work is available.

    def get_task(self):
        """
        @brief Retrieves the next available task from the WorkPool's queue.
        @return The next task tuple if available, otherwise None.
        Functional Utility: Provides a thread-safe way for workers to get tasks.
        """
        # Block Logic: Checks if there are more tasks to process.
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index] # Get the current task.
            self.current_task_index = self.current_task_index + 1 # Move to the next task index.

            # Block Logic: If all tasks have been distributed, clear `work_to_do` and set `tasks_finish`.
            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear()   # Clear event as no more tasks are immediately available.
                self.tasks_finish.set()   # Signal that all tasks for this batch are being processed.

            return task
        else:
            return None # No more tasks.

    def close(self):
        """
        @brief Gracefully shuts down the WorkPool and all its worker threads.
        Functional Utility: Ensures all pending tasks are finished and worker threads
                            are properly terminated.
        """
        self.tasks = []                        # Clear any remaining tasks.
        self.current_task_index = len(self.tasks) # Mark all tasks as "processed" for graceful shutdown.

        # Block Logic: Signal workers that there is no more work to do, allowing them to exit.
        self.work_to_do.set()

        # Block Logic: Wait for all worker threads to fully terminate.
        for worker in self.workers:
            worker.join()


class Worker(Thread):
    """
    @class Worker
    @brief A thread that executes tasks provided by its parent WorkPool.
           It is responsible for running scripts on specified data locations,
           gathering data from neighbors, and updating results, all while
           respecting global locks for data consistency.
    """

    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        """
        @brief Initializes a Worker thread.
        @param work_pool: Reference to the parent WorkPool instance.
        @param lock_get_task: A Lock for thread-safe access to task retrieval logic.
        @param work_to_do: An Event to signal when tasks are available for workers.
        @param lock_locations: Shared list of global Locks for data locations.
        """
        Thread.__init__(self)
        self.work_pool = work_pool         # Reference to the parent WorkPool.
        self.lock_get_task = lock_get_task # Lock for getting tasks.
        self.work_to_do = work_to_do       # Event for work availability.
        self.lock_locations = lock_locations # Shared global data locks.

    def run(self):
        """
        @brief The main execution loop for the Worker thread.
        Functional Utility: Continuously picks up tasks, executes scripts with proper
                            global locking for data, and updates device data.
        """
        # Block Logic: The main loop for the worker thread. It continuously tries to acquire
        #              the task retrieval lock and waits for work to be available.
        while True:
            # Block Logic: Acquires a lock to ensure only one worker attempts to get a task at a time.
            self.lock_get_task.acquire()

            # Block Logic: Waits until the `work_to_do` event is set, signaling that tasks are available.
            self.work_to_do.wait()

            # Functional Utility: Retrieves the next task from the WorkPool.
            task = self.work_pool.get_task()

            # Block Logic: Releases the task retrieval lock.
            self.lock_get_task.release()

            # If `get_task` returns None, it means the WorkPool is closing, so the worker exits.
            if task is None:
                break

            # Unpack the task components.
            script = task[0]
            location = task[1]
            neighbours = task[2]
            self_device = task[3]

            # Block Logic: Acquires the global lock specific to the data `location`.
            #              This ensures exclusive access to data at this `location` across all devices/workers.
            self.lock_locations[location].acquire()

            script_data = [] # List to aggregate data for the current script execution.

            # Block Logic: Gathers data from all neighboring devices for the specified `location`.
            # Invariant: `script_data` will contain relevant data from neighbors.
            for device in neighbours:
                data = device.get_data(location) # Retrieves data using the neighbor's local get_data.
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Gathers data from the current device (`self_device`) for the specified `location`.
            data = self_device.get_data(location) # Retrieves data using the current device's local get_data.
            if data is not None:
                script_data.append(data)

            # Block Logic: If any data was collected, execute the script and update device(s) data.
            if script_data != []:
                # Executes the assigned script with the collected data.
                result = script.run(script_data)

                # Block Logic: Updates the data on all neighboring devices with the script's result.
                for device in neighbours:
                    device.set_data(location, result)
                
                # Block Logic: Updates the data on the current device (`self_device`) with the script's result.
                self_device.set_data(location, result)

            # Releases the global lock for the current data `location`.
            self.lock_locations[location].release()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief Manages the execution lifecycle of scripts for a single device within the
           distributed simulation. It interacts with the supervisor, its device's
           `WorkPool`, and the global conditional barrier.
    """

    def __init__(self, device, barrier, tasks_finish):
        """
        @brief Initializes the DeviceThread.
        @param device: The Device instance this thread is managing.
        @param barrier: A reference to the global `ReusableBarrierCond` for inter-device synchronization.
        @param tasks_finish: An `Event` that the device's `WorkPool` sets upon task completion.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device             # Reference to the associated Device instance.
        self.barrier = barrier           # Reference to the global barrier.
        self.tasks_finish = tasks_finish # Event to wait on for WorkPool task completion.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Continuously synchronizes with other devices, fetches neighbors,
                            submits scripts to its `WorkPool`, waits for completion,
                            and prepares for the next timepoint.
        """
        # Block Logic: The main loop for the device's operational thread.
        # Invariant: The device continuously processes time steps until a shutdown signal is received.
        while True:
            # Functional Utility: Synchronizes all devices at the reusable conditional barrier.
            #                     All devices must reach this point before any can proceed to the next time step.
            self.barrier.wait()

            # Block Logic: Fetches the current neighbors of the device from the supervisor.
            #              The supervisor determines the current set of active neighbors or signals shutdown.
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                self.device.work_pool.close() # Signal this device's work pool to shut down its workers.
                break # Exit the main device loop.

            # Block Logic: Waits until the `script_received` event is set, indicating that new scripts
            #              for the current timepoint have been assigned to this device.
            self.device.script_received.wait()

            tasks = [] # List to temporarily hold tasks for this timepoint.

            # Block Logic: Prepares tasks from the assigned scripts, including necessary context for workers.
            # Invariant: `tasks` list is populated with tuples (script, location, neighbours, self_device).
            for (script, location) in self.device.scripts:
                tasks.append((script, location, neighbours, self.device))

            # Block Logic: If there are tasks for this device, submit them to its WorkPool.
            if tasks != []:
                self.device.work_pool.set_tasks(tasks) # Assigns the new batch of tasks to the WorkPool.

                # Blocks until the WorkPool signals that all tasks submitted by this device are finished.
                self.tasks_finish.wait()

                # Resets the `tasks_finish` event for the next cycle.
                self.tasks_finish.clear()

            # Resets the `script_received` event for the next cycle.
            self.device.script_received.clear()
