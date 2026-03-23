"""
@file device.py
@brief Implements a distributed device simulation framework with a shared work pool
       for executing scripts and a reusable barrier for global synchronization.
       This module orchestrates data processing across multiple simulated devices
       that can interact with their neighbors.

Algorithm:
- **Device Initialization:** Each `Device` holds local sensor data, scripts, and references
  to global synchronization primitives (locks, barrier) and a shared `WorkPool`.
- **Shared Resource Setup:** The first `Device` to call `setup_devices` initializes
  global locks for data locations, a `ReusableBarrier` for inter-device synchronization,
  and a `WorkPool` for executing scripts. Other devices then link to these shared instances.
- **Script Assignment & Execution Flow:**
    1. Scripts are assigned to individual `Device` instances, along with a target data `location`.
    2. Each `DeviceThread` waits for scripts to be assigned.
    3. Assigned scripts are submitted as tasks to the global `WorkPool`.
    4. `WorkerThread`s from the `WorkPool` pick up tasks. For each task:
        a. A global lock for the specific data `location` is acquired to ensure exclusive access.
        b. Data is gathered from the assigned device and its neighbors for that `location`.
        c. The script's `run` method is executed with the collected data.
        d. The results are used to update data on the device and its neighbors.
        e. The global lock for the `location` is released.
    5. The `DeviceThread` waits for all its submitted scripts to be processed by the `WorkPool`.
    6. All `DeviceThread`s then synchronize at a `ReusableBarrier` before proceeding to the next time step.
- **Shutdown:** The `WorkPool` is explicitly ended, and device threads are joined.

Time Complexity:
- The overall time complexity is highly dependent on:
    - The number of devices (N).
    - The number and complexity of scripts (S).
    - The network topology (number of neighbors per device).
    - The number of worker threads in the `WorkPool`.
- Script execution is parallelized by the `WorkPool`. Synchronization points (`barrier.wait()`, `WorkPool.finish_work()`) can be bottlenecks.

Space Complexity:
- O(N_devices * (D + S)) for storing device-specific sensor data (D) and scripts (S).
- Additional overhead for:
    - Global locks (proportional to unique data locations).
    - `ReusableBarrier` and `WorkPool` internal structures.
    - Thread stacks for `DeviceThread`s and `WorkerThread`s.
"""

from threading import Event, Thread, Lock, Semaphore # Import necessary threading primitives.
from pool import WorkPool                           # Import the custom WorkPool class.
from reusable_barrier import ReusableBarrier        # Import the custom ReusableBarrier class.

class Device(object):
    """
    @class Device
    @brief Represents a single simulated device in the distributed system.
           Manages its local sensor data, assigned scripts, and synchronization mechanisms.
           It interacts with a central supervisor and a shared work pool.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id: Unique identifier for this device.
        @param sensor_data: A dictionary containing sensor readings/data specific to this device.
        @param supervisor: The central supervisor object responsible for managing all devices.
        """
        self.device_id = device_id       # Unique identifier for the device.
        self.sensor_data = sensor_data   # Local sensor data, a dictionary (e.g., {'location_id': data_value}).
        self.supervisor = supervisor     # Reference to the supervisor object.

        self.script_received = Event()   # Event to signal when new scripts have been assigned.
        self.scripts = []                # List to store assigned scripts (script_obj, location_id).

        self.timepoint_done = Event()    # Event for internal device synchronization (less critical with barrier).
        self.other_devs = []             # List of all other devices in the simulation.
        self.slock = Lock()              # Local lock to protect access to `self.sensor_data`.

        self.barrier = None              # Reference to the shared ReusableBarrier for global synchronization.
        self.process = Event()           # Event for internal process control (usage might be limited).

        self.global_thread_pool = None   # Reference to the shared WorkPool for script execution.
        self.glocks = {}                 # Reference to the shared dictionary of global locks per data location.

        self.thread = DeviceThread(self) # Create and start the dedicated thread for this device.
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives (barrier, work pool, and global locks)
               among all devices in the simulation. This is typically called once by one device.
        @param devices: A list of all Device instances in the simulation.
        Functional Utility: Ensures all devices share the same global resources for consistent
                            concurrent operation and synchronization.
        """
        self.other_devs = devices # Store the list of all devices.

        # Block Logic: The first device (e.g., device with smallest ID, or first in list)
        #              takes responsibility for initializing the global shared resources.
        # Invariant: After this block, all devices will share the same `glocks`, `barrier`,
        #            and `global_thread_pool` instances.
        if self.device_id == self.other_devs[0].device_id: # Check if this is the "master" device for setup.
            locks = {} # Local dictionary to build the global locks.
            for loc in self.sensor_data:
                locks[loc] = Lock() # Create a lock for each data location.
            dev_cnt = len(devices) # Count of all devices.
            self.glocks = locks    # This device's `glocks` now points to the new global locks.
            self.barrier = ReusableBarrier(dev_cnt) # Initialize the reusable barrier for `dev_cnt` participants.
            self.global_thread_pool = WorkPool(16) # Initialize the shared work pool with 16 worker threads.
        else:
            # Other devices connect to the shared resources initialized by the "master" device.
            for loc in self.sensor_data:
                # Ensure each location in this device's sensor_data has an associated global lock.
                # If a location is only present on a "slave" device, its lock might not be created by the master.
                # This could be a potential bug if master doesn't have all locations present across all slaves.
                if loc not in self.other_devs[0].glocks:
                    self.other_devs[0].glocks[loc] = Lock()
            self.glocks = self.other_devs[0].glocks # Link to the master's global locks.
            self.global_thread_pool = self.other_devs[0].global_thread_pool # Link to the master's work pool.
            self.barrier = self.other_devs[0].barrier # Link to the master's barrier.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific data location.
        @param script: The script object to be executed (expected to have a `run` method).
        @param location: The data location (key) the script operates on.
        Functional Utility: Queues scripts for processing by the device's thread.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
        else:
            # If `script` is None, it acts as a signal to the DeviceThread that no more scripts
            # are coming for the current timepoint, and it should proceed.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location, protected by a local lock.
        @param location: The key corresponding to the data location.
        @return The data at the specified location, or None if the location does not exist.
        Functional Utility: Provides thread-safe access to the device's local sensor data.
        """
        ret = None
        # Block Logic: Acquires a local lock (`self.slock`) to ensure exclusive access
        #              when reading `self.sensor_data`.
        with self.slock:
            if location in self.sensor_data:
                ret = self.sensor_data[location]
        return ret

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a specific location, protected by a local lock.
        @param location: The key corresponding to the data location.
        @param data: The new data value to set.
        Functional Utility: Provides thread-safe modification of the device's local sensor data.
        """
        # Block Logic: Acquires a local lock (`self.slock`) to ensure exclusive access
        #              when writing to `self.sensor_data`.
        with self.slock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's thread.
        Functional Utility: Ensures proper termination and cleanup of the `DeviceThread`.
        """
        self.thread.join() # Waits for the DeviceThread to complete its execution.


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief Manages the execution lifecycle of scripts for a single device within the
           distributed simulation. It interacts with the supervisor, work pool, and barrier.
    """
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device: The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device # Reference to the associated Device instance.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Continuously fetches neighbors, waits for script assignments,
                            submits scripts to the shared work pool, waits for script
                            completion, and synchronizes globally at a barrier.
        """
        # Block Logic: The main loop for the device's operational thread.
        # Invariant: The device continuously processes time steps until a shutdown signal is received.
        while True:
            # Block Logic: Fetches the current neighbors of the device from the supervisor.
            #              The supervisor determines the current set of active neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # If no neighbors are returned (e.g., simulation has ended or supervisor signaled shutdown),
                # this device needs to signal the global thread pool to end if it's the master device.
                if self.device.device_id is self.device.other_devs[0].device_id:
                    self.device.global_thread_pool.end() # Signal the work pool to shut down its workers.
                break # Exit the main device loop.

            # Block Logic: Waits until the `script_received` event is set.
            #              This event is typically set by the supervisor when all scripts
            #              for the current timepoint have been assigned to this device.
            self.device.script_received.wait()

            # Block Logic: Submits all assigned scripts to the global thread pool for execution.
            # Invariant: After this loop, all scripts from `self.device.scripts` have been
            #            enqueued as tasks in the `global_thread_pool`.
            for (script, location) in self.device.scripts:
                self.device.global_thread_pool.work((self.device, # The device instance (for data access).
                                                      script,     # The script to run.
                                                      location,   # The data location it operates on.
                                                      neighbours)) # The device's neighbors.

            # Functional Utility: Blocks until all work submitted by this device to the global
            #                     thread pool has been completed by worker threads.
            self.device.global_thread_pool.finish_work()
            self.device.script_received.clear() # Resets the event for the next timepoint.
            
            # Functional Utility: Synchronizes all devices at the reusable barrier.
            #                     All devices must reach this point before any can proceed.
            self.device.barrier.wait()


# Moved from pool.py for context, as requested by the user's prompt.
# It is important to remember that these classes are likely defined in separate files.
# For this context, they are placed here for a complete understanding of the system.


class WorkerThread(Thread):
    """
    @class WorkerThread
    @brief A thread that executes tasks from the shared WorkPool.
           Each task involves running a script on a device's data location
           and interacting with its neighbors, all while ensuring data consistency
           via global locks.
    """
    def __init__(self, i, parent_work_pool):
        """
        @brief Initializes a WorkerThread.
        @param i: A unique identifier for this worker thread.
        @param parent_work_pool: Reference to the WorkPool that owns this thread.
        """
        Thread.__init__(self, name="WorkerThread%d" % i)
        self.pool = parent_work_pool # Reference to the parent WorkPool.

    def run(self):
        """
        @brief The main execution loop for the WorkerThread.
        Functional Utility: Continuously picks up tasks from the WorkPool,
                            executes scripts with proper locking, and updates data.
        """
        # Block Logic: The main loop for the worker thread. It continuously tries to acquire
        #              a task semaphore, indicating available work.
        # Invariant: The worker continues to process tasks until the WorkPool is signaled to stop.
        while True:
            self.pool.task_sign.acquire() # Blocks until a task is available (semaphore count > 0).
            if self.pool.stop:
                break # If the pool is stopping, exit the worker thread loop.
            
            current_task = (None, None, None, None) # Initialize current_task.
            # Block Logic: Acquires a lock to safely access and modify the shared `tasks_list`.
            with self.pool.task_lock:
                task_count = len(self.pool.tasks_list)
                
                if task_count > 0:
                    current_task = self.pool.tasks_list[0] # Get the next task.
                    self.pool.tasks_list = self.pool.tasks_list[1:] # Remove the task from the list.
                
                # Block Logic: If the task list becomes empty after popping a task,
                #              set the `no_tasks` event to signal that no tasks remain.
                if task_count == 1:
                    self.pool.no_tasks.set()

            # Block Logic: If a valid task was retrieved, process it.
            if current_task is not (None, None, None, None): # Check if current_task is a valid task.
                (current_device, script, location, neighbourhood) = current_task
                
                # Block Logic: Acquires the global lock specific to the data `location`.
                #              This ensures exclusive access to data at this location across all devices/workers.
                with current_device.glocks[location]:
                    common_data = [] # List to aggregate data for the script.
                    
                    # Block Logic: Gathers data from all neighboring devices for the specified `location`.
                    for neighbour in neighbourhood:
                        data = neighbour.get_data(location) # Retrieves data using the neighbor's local lock.
                        if data is not None:
                            common_data.append(data)
                    
                    # Block Logic: Gathers data from the current device itself for the specified `location`.
                    data = current_device.get_data(location) # Retrieves data using the current device's local lock.
                    if data is not None:
                        common_data.append(data)

                    # Block Logic: If any data was collected, execute the script and update device(s) data.
                    if common_data != []:
                         # Executes the assigned script with the collected data.
                        result = script.run(common_data)
                        
                        # Block Logic: Updates the data on all neighboring devices with the script's result.
                        for neighbour in neighbourhood:
                            neighbour.set_data(location, result)
                        
                        # Block Logic: Updates the data on the current device with the script's result.
                        current_device.set_data(location, result)


class WorkPool(object):
    """
    @class WorkPool
    @brief Manages a pool of worker threads (`WorkerThread`) to execute tasks asynchronously.
           Provides mechanisms to submit tasks, wait for their completion, and gracefully shut down.
    """
    def __init__(self, size):
        """
        @brief Initializes the WorkPool with a specified number of worker threads.
        @param size: The number of worker threads to create in the pool.
        """
        self.size = size # Number of worker threads.

        self.tasks_list = []      # List to hold tasks submitted to the pool.
        self.task_lock = Lock()   # Lock to protect concurrent access to `tasks_list`.
        self.task_sign = Semaphore(0) # Semaphore to signal worker threads about available tasks.
        self.no_tasks = Event()   # Event to signal when there are no tasks left in the queue.
        self.no_tasks.set()       # Initially set, as there are no tasks.
        self.stop = False         # Flag to indicate if the work pool should shut down.

        self.workers = []         # List to hold worker thread instances.
        # Block Logic: Creates and starts the specified number of worker threads.
        for i in range(self.size): # Using range instead of xrange for Python 3 compatibility.
            worker = WorkerThread(i, self)
            self.workers.append(worker)

        for worker in self.workers:
            worker.start() # Start each worker thread.

    def work(self, task):
        """
        @brief Submits a new task to the work pool.
        @param task: The task to be executed (e.g., a tuple containing device, script, location, neighbors).
        Functional Utility: Adds a task to the queue and notifies a waiting worker thread.
        """
        # Block Logic: Acquires a lock to safely add a task to the shared `tasks_list`.
        with self.task_lock:
            self.tasks_list.append(task) # Add the new task.
            self.task_sign.release()     # Increment the semaphore, notifying a worker thread.
            # If `no_tasks` was set (meaning no tasks were present), clear it as a new task is added.
            if self.no_tasks.is_set():
                self.no_tasks.clear()

    def finish_work(self):
        """
        @brief Blocks until all tasks currently in the queue have been processed.
        Functional Utility: Provides a synchronization point, ensuring all submitted
                            work is completed before proceeding.
        """
        self.no_tasks.wait() # Blocks until the `no_tasks` event is set (i.e., queue is empty).

    def end(self):
        """
        @brief Gracefully shuts down the work pool and all its worker threads.
        Functional Utility: Ensures all pending tasks are finished and worker threads
                            are properly terminated.
        """
        self.finish_work() # First, wait for all existing tasks to complete.
        self.stop = True   # Set the stop flag to signal workers to terminate.
        # Block Logic: Release the semaphore `size` times to wake up all worker threads,
        #              allowing them to see the `stop` flag and exit.
        for thread in self.workers:
            self.task_sign.release()
        # Block Logic: Wait for all worker threads to fully terminate.
        for thread in self.workers:
            thread.join()
