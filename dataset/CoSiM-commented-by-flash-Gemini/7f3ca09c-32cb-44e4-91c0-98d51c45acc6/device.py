


from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single computational device in a simulated distributed system.
    Each device has an ID, sensor data, and communicates with a supervisor.
    It can receive and execute scripts, leveraging a work pool for concurrent processing
    and synchronizing with other devices using a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device object.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing sensor readings or local data,
                                 keyed by location.
            supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when new scripts have been assigned to this device.
        self.script_received = Event()
        self.scripts = [] # List to store (script, location) tuples assigned to this device.
        # Event to signal when all scripts for the current timepoint are done. (Not explicitly used directly by Device, but WorkPool uses it)
        self.timepoint_done = Event() 
        # The main thread for this device; will be initialized in setup_devices.
        self.thread = None
        # The WorkPool instance responsible for executing scripts; will be initialized in setup_devices.
        self.work_pool = None

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the shared barrier, per-location locks, and work pools for all devices.
        Only device with ID 0 initializes these shared resources and starts the DeviceThreads.

        Args:
            devices (List[Device]): A list of all Device instances in the system.
        """
        # Block Logic: Only the device with ID 0 initializes shared resources (barrier, locks)
        # and starts the main threads for all devices.
        if self.device_id == 0:
            # Initializes a ReusableBarrierCond (condition-based barrier) with the total number of devices.
            barrier = ReusableBarrierCond(len(devices))
            lock_locations = []

            # Block Logic: Initializes a list of Locks, one for each unique location across all devices.
            # This ensures thread-safe access to sensor data at specific locations.
            # Using xrange assumes location keys are contiguous integers starting from 0.
            for device in devices:
                for _ in xrange(len(device.sensor_data)): # Iterates through sensor_data keys to determine count.
                    lock_locations.append(Lock())

            # Block Logic: Initializes a WorkPool and a DeviceThread for each device.
            for device in devices:
                tasks_finish = Event() # Event to signal when all tasks in a WorkPool are finished.
                device.work_pool = WorkPool(tasks_finish, lock_locations) # Assigns a WorkPool to each device.

                # Initializes and starts the main operational thread for each device.
                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a specific location.
        If script is None, it signals that new scripts have been received, but no
        actual script was passed (e.g., to trigger barrier synchronization).

        Args:
            script (Script or None): The script object to execute, or None for signaling.
            location (int): The logical location (index) associated with the script's execution.
        """
        # Conditional Logic: If a script object is provided, add it to the device's script list.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If script is None, it signals that the script assignment phase for this
            # timepoint is complete.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The key (index) for the sensor data.

        Returns:
            Any: The data associated with the location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data \
                else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location. Access to `sensor_data` is
        not explicitly protected by a lock here, relying on external mechanisms
        (e.g., `lock_locations` in Worker) for thread safety.

        Args:
            location (int): The key (index) for the sensor data.
            data (Any): The new data to set for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its operational thread, ensuring all
        tasks are completed before the program exits.
        """
        self.thread.join()


class WorkPool(object):
    """
    Manages a pool of worker threads to execute tasks concurrently.
    It distributes tasks to available workers and provides synchronization
    mechanisms for task completion.
    """

    def __init__(self, tasks_finish, lock_locations):
        """
        Initializes a WorkPool.

        Args:
            tasks_finish (Event): An Event object that is set when all tasks
                                  assigned to this work pool are completed.
            lock_locations (List[Lock]): A list of locks, where each lock protects
                                         data at a specific location across devices.
        """
        self.workers = [] # List to hold Worker thread instances.
        self.tasks = []   # List of tasks (scripts) to be executed.
        self.current_task_index = 0 # Index of the next task to be assigned.
        self.lock_get_task = Lock() # Lock to protect `current_task_index` and `tasks` access.
        self.work_to_do = Event()   # Event to signal workers that there are tasks to process.
        self.tasks_finish = tasks_finish # Event to signal completion of all tasks in the pool.
        self.lock_locations = lock_locations # Shared list of location locks.
        self.max_num_workers = 8 # Maximum number of worker threads in the pool.

        # Block Logic: Creates and starts the worker threads.
        for i in xrange(self.max_num_workers):
            self.workers.append(Worker(self, self.lock_get_task, \
                    self.work_to_do, self.lock_locations))
            self.workers[i].start()

    def set_tasks(self, tasks):
        """
        Assigns a new list of tasks to the work pool.

        Args:
            tasks (List[Tuple]): A list of tasks, where each task is a tuple
                                 (script, location, neighbours, self_device).
        """
        self.tasks = tasks
        self.current_task_index = 0

        # Signals to workers that there is work to be done.
        self.work_to_do.set()

    def get_task(self):
        """
        Retrieves the next available task from the pool. This method is called by workers.

        Returns:
            Tuple or None: The next task tuple, or None if no more tasks are available.
        """
        # Conditional Logic: Checks if there are more tasks to distribute.
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index]

            self.current_task_index = self.current_task_index + 1

            # Conditional Logic: If this was the last task, clear `work_to_do` and set `tasks_finish`.
            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear()
                self.tasks_finish.set() # Signals that all tasks have been handed out.

            return task
        else:
            return None

    def close(self):
        """
        Signals the work pool to shut down. This clears any remaining tasks
        and waits for all worker threads to terminate.
        """
        self.tasks = [] # Clear remaining tasks.
        self.current_task_index = len(self.tasks) # Indicate no more tasks.

        # Signal workers to wake up and find no more tasks, leading them to exit.
        self.work_to_do.set()

        # Block Logic: Joins all worker threads to ensure their termination.
        for worker in self.workers:
            worker.join()

class Worker(Thread):
    """
    A worker thread that continuously fetches tasks from a WorkPool,
    executes scripts, gathers data from devices, and updates shared data.
    """

    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        """
        Initializes a Worker thread.

        Args:
            work_pool (WorkPool): The WorkPool instance from which to get tasks.
            lock_get_task (Lock): A lock to synchronize access to `work_pool.get_task()`.
            work_to_do (Event): An Event object that signals when there are tasks in the work pool.
            lock_locations (List[Lock]): A list of locks for per-location data access.
        """
        Thread.__init__(self)
        self.work_pool = work_pool
        self.lock_get_task = lock_get_task
        self.work_to_do = work_to_do
        self.lock_locations = lock_locations

    def run(self):
        """
        The main execution loop for the Worker thread.
        It repeatedly waits for tasks, acquires a task from the work pool,
        executes the script (gathering data from neighbors and its own device),
        and updates data, ensuring thread safety with location-specific locks.
        """
        while True:
            # Block Logic: Acquire lock before attempting to get a task to prevent race conditions.
            self.lock_get_task.acquire()

            # Block Logic: Wait until the WorkPool signals that there's work to do.
            self.work_to_do.wait()

            # Get a task from the work pool.
            task = self.work_pool.get_task()

            # Release the lock after getting the task.
            self.lock_get_task.release()

            # Conditional Logic: If no task is returned (signaling shutdown), break the loop.
            if task is None:
                break

            # Unpack the task details.
            script = task[0]
            location = task[1]
            neighbours = task[2]
            self_device = task[3]

            # Block Logic: Acquire the location-specific lock to protect data at this location.
            self.lock_locations[location].acquire()

            script_data = [] # Data collected for the script.

            # Block Logic: Collects data from neighboring devices for the specified location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Collects data from the current device for the specified location.
            data = self_device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Conditional Logic: If there is any data, execute the script.
            if script_data != []:
                # Executes the script with the collected data.
                result = script.run(script_data)

                # Block Logic: Updates the data on neighboring devices with the script's result.
                for device in neighbours:
                    device.set_data(location, result)
                
                # Block Logic: Updates the data on the current device with the script's result.
                self_device.set_data(location, result)

            # Release the location-specific lock.
            self.lock_locations[location].release()



class DeviceThread(Thread):
    """
    The main operational thread for a Device. It drives the device's behavior,
    synchronizing with other devices via a barrier, delegating script execution
    to a WorkPool, and handling the timepoint lifecycle.
    """

    def __init__(self, device, barrier, tasks_finish):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The Device instance that this thread controls.
            barrier (ReusableBarrierCond): The shared barrier for inter-device synchronization.
            tasks_finish (Event): An Event object from the device's WorkPool,
                                  signaling when all tasks are complete.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.tasks_finish = tasks_finish

    def run(self):
        """
        The main execution loop for the DeviceThread.
        It repeatedly waits at the barrier, gets neighbors, waits for scripts to be
        assigned, sets tasks to the WorkPool, waits for WorkPool completion,
        and then clears events for the next cycle.
        """
        while True:
            # Block Logic: Waits at the shared barrier for all DeviceThreads to reach this point.
            # This ensures all devices are ready before starting a new timepoint.
            self.barrier.wait()

            # Fetches the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Conditional Logic: If no neighbors are returned (e.g., system shutdown),
            # close the WorkPool and exit the loop.
            if neighbours is None:
                self.device.work_pool.close() # Signal workers to shut down.
                break

            # Block Logic: Waits for scripts to be assigned to this device for the current timepoint.
            self.device.script_received.wait()

            tasks = [] # List to accumulate tasks for the WorkPool.

            # Block Logic: Prepares tasks from the device's assigned scripts.
            for (script, location) in self.device.scripts:
                tasks.append((script, location, neighbours, self.device))

            # Conditional Logic: If there are tasks, assign them to the WorkPool.
            if tasks != []:
                self.device.work_pool.set_tasks(tasks) # Assigns tasks to the WorkPool.

                # Block Logic: Waits until all tasks in the WorkPool are finished.
                self.tasks_finish.wait()

                # Clear the event for the next cycle.
                self.tasks_finish.clear()

            # Clear the script_received event for the next timepoint.
            self.device.script_received.clear()

