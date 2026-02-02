

"""
This module implements a simulated device with threading capabilities for sensor data processing
in a distributed environment. It includes mechanisms for inter-device synchronization and task management.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a simulated device capable of processing sensor data and interacting with a supervisor.
    Manages its own scripts, sensor data, and threading for concurrent operation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding sensor data, keyed by location.
            supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal when new scripts are assigned.
        self.scripts = []  # List to store assigned scripts, each with its target location.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's processing.
        self.thread = None  # Placeholder for the DeviceThread instance.
        self.work_pool = None # Placeholder for the WorkPool instance managing script execution.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Orchestrates the setup of all simulated devices, initializing barriers and work pools.
        This function is intended to be called by a master device (device_id == 0).

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        # Block Logic: Initializes shared synchronization primitives and work pools for all devices.
        # Invariant: Ensures that synchronization and work management infrastructure is set up once.
        if self.device_id == 0:
            # Initializes a reusable barrier for synchronizing all devices at specific timepoints.
            barrier = ReusableBarrierCond(len(devices))
            lock_locations = []

            # Block Logic: Populates a list of locks, one for each sensor data location across all devices.
            # These locks prevent race conditions during concurrent updates to sensor data.
            for device in devices:
                for _ in xrange(len(device.sensor_data)):
                    lock_locations.append(Lock())

            # Block Logic: Configures each device with its own work pool and a dedicated thread for execution.
            # Invariant: Each device is assigned a unique thread and a work pool for managing its tasks.
            for device in devices:
                # Event to signal when all tasks assigned to a device's work pool have finished.
                tasks_finish = Event()
                # Initializes a WorkPool for the device to manage concurrent script execution.
                device.work_pool = WorkPool(tasks_finish, lock_locations)

                # Creates and starts a dedicated thread for the device's operational logic.
                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific sensor data location.

        Args:
            script (Script): The script object to be executed.
            location (int): The identifier for the sensor data location the script targets.
        """
        # Block Logic: Appends the script and its target location to the device's script queue.
        # If no script is provided, it signals that script assignment is complete for the current timepoint.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set() # Signals that all scripts for the current timepoint have been received.

    def get_data(self, location):
        """
        Retrieves sensor data from a specified location on this device.

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if the location does not exist.
        """
        return self.sensor_data[location] if location in self.sensor_data \
                else None

    def set_data(self, location, data):
        """
        Sets sensor data at a specified location on this device.

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new data value to set.
        """
        # Block Logic: Updates the sensor data at the specified location if it exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device's operational thread.
        """
        # Block Logic: Waits for the device's thread to complete its execution before proceeding with shutdown.
        self.thread.join()

class WorkPool(object):
    """
    Manages a pool of worker threads responsible for executing scripts.
    Distributes tasks among workers and synchronizes their execution.
    """

    def __init__(self, tasks_finish, lock_locations):
        """
        Initializes a new WorkPool instance.

        Args:
            tasks_finish (Event): An Event object to signal when all tasks in the pool are finished.
            lock_locations (list): A list of Lock objects, one for each sensor data location.
        """
        self.workers = []  # List to hold Worker thread instances.
        self.tasks = []  # Queue of tasks to be processed.
        self.current_task_index = 0  # Index of the next task to be assigned.
        self.lock_get_task = Lock()  # Lock to ensure exclusive access when workers retrieve tasks.
        self.work_to_do = Event()  # Event to signal when new tasks are available for workers.
        self.tasks_finish = tasks_finish  # Event to signal when all tasks in the pool are finished.
        self.lock_locations = lock_locations  # Locks to protect access to sensor data locations.
        self.max_num_workers = 8  # Maximum number of worker threads in the pool.

        # Block Logic: Initializes and starts a fixed number of Worker threads.
        # Invariant: Ensures that a predefined number of worker threads are ready to process tasks.
        for i in xrange(self.max_num_workers):
            self.workers.append(Worker(self, self.lock_get_task, \
                    self.work_to_do, self.lock_locations))
            self.workers[i].start()

    def set_tasks(self, tasks):
        """
        Assigns a new set of tasks to the work pool and signals workers to begin processing.

        Args:
            tasks (list): A list of tasks (script, location, neighbours, self_device) to be executed.
        """
        self.tasks = tasks
        self.current_task_index = 0

        # Signals all waiting workers that new work is available.
        self.work_to_do.set()

    def get_task(self):
        """
        Retrieves the next available task from the pool.

        Returns:
            tuple: A task tuple (script, location, neighbours, self_device) or None if no tasks remain.
        """
        # Block Logic: Checks if there are pending tasks. If so, it retrieves the next task
        # and updates the task index. If all tasks are assigned, it clears the 'work_to_do'
        # event and sets the 'tasks_finish' event.
        # Invariant: Ensures atomic retrieval of tasks and proper state management of task completion.
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index]

            self.current_task_index = self.current_task_index + 1

            # Checks if all tasks have been assigned; if so, updates the state of work availability.
            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear()
                self.tasks_finish.set()

            return task
        else:
            return None

    def close(self):
        """
        Shuts down the work pool, ensuring all worker threads terminate gracefully.
        """
        self.tasks = []
        # Sets the current task index beyond the list length to signal no more tasks.
        self.current_task_index = len(self.tasks)

        # Signals workers one last time to wake up and discover there are no more tasks.
        self.work_to_do.set()

        # Block Logic: Waits for each worker thread to complete its execution before continuing.
        for worker in self.workers:
            worker.join()

class Worker(Thread):
    """
    A worker thread that continuously fetches and executes scripts from a WorkPool.
    Handles synchronization for task retrieval and sensor data access.
    """

    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        """
        Initializes a new Worker thread.

        Args:
            work_pool (WorkPool): The WorkPool instance from which to get tasks.
            lock_get_task (Lock): A Lock for synchronizing access to the WorkPool's task queue.
            work_to_do (Event): An Event to wait on until new tasks are available.
            lock_locations (list): A list of Lock objects to protect sensor data locations.
        """
        Thread.__init__(self)
        self.work_pool = work_pool
        self.lock_get_task = lock_get_task
        self.work_to_do = work_to_to  # Event to wait on for new work.
        self.lock_locations = lock_locations  # Locks for sensor data.

    def run(self):
        """
        The main execution loop for the worker thread. Continuously retrieves and processes tasks.
        """
        # Block Logic: The worker's main loop; it continuously attempts to acquire tasks.
        # Invariant: The worker remains active, waiting for work and processing tasks until signaled to stop.
        while True:
            # Acquires a lock to ensure exclusive access to the shared task queue in the WorkPool.
            self.lock_get_task.acquire()

            # Waits until the 'work_to_do' event is set, indicating that tasks are available.
            self.work_to_do.wait()

            task = self.work_pool.get_task()

            # Releases the lock on the task queue.
            self.lock_get_task.release()

            # Block Logic: Checks if a task was successfully retrieved.
            # If no task is available (None), the worker terminates its loop.
            if task is None:
                break

            # Decomposes the task tuple into its constituent components.
            script = task[0]
            location = task[1]
            neighbours = task[2]
            self_device = task[3]

            # Acquires a lock specific to the sensor data location to prevent concurrent modifications.
            self.lock_locations[location].acquire()

            script_data = []

            # Block Logic: Collects sensor data from neighboring devices and the current device for the script.
            # Invariant: All available sensor data for the specified location is gathered before script execution.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Retrieves data from the current device's sensor at the specified location.
            data = self_device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script if sensor data is available and updates sensor data across devices.
            # Invariant: If a script runs, its results are consistently applied to the relevant sensor locations.
            if script_data != []:
                # Executes the assigned script with the collected sensor data.
                result = script.run(script_data)

                # Propagates the script's result to the corresponding sensor location on neighboring devices.
                for device in neighbours:
                    device.set_data(location, result)
                
                # Updates the current device's sensor data with the script's result.
                self_device.set_data(location, result)

            # Releases the lock on the sensor data location after updates are complete.
            self.lock_locations[location].release()



class DeviceThread(Thread):
    """
    Manages the lifecycle and operational logic of a single simulated device.
    Synchronizes with other devices using a barrier and delegates script execution to a WorkPool.
    """

    def __init__(self, device, barrier, tasks_finish):
        """
        Initializes a new DeviceThread.

        Args:
            device (Device): The Device instance this thread controls.
            barrier (ReusableBarrierCond): A synchronization barrier for coordinating with other devices.
            tasks_finish (Event): An Event object from the device's WorkPool to signal task completion.
        """
        # Initializes the Thread with a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier  # Synchronization barrier for timepoint coordination.
        self.tasks_finish = tasks_finish  # Event for signaling when work pool tasks are finished.

    def run(self):
        """
        The main execution loop for the device thread.
        Synchronizes at barriers, retrieves neighbors, assigns scripts to its work pool,
        and waits for script execution to complete.
        """
        # Block Logic: Main operational loop for the device.
        # Invariant: The device continuously processes timepoints, synchronizes, executes scripts, and manages its work pool.
        while True:
            # Waits at the barrier to synchronize with all other devices before starting a new timepoint.
            self.barrier.wait()

            # Retrieves information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Checks if there are no more neighbors, signaling the end of the simulation.
            # If so, it closes its work pool and terminates the thread.
            if neighbours is None:
                self.device.work_pool.close()
                break

            # Waits for new scripts to be assigned to the device for the current timepoint.
            self.device.script_received.wait()

            tasks = []

            # Block Logic: Prepares tasks for the work pool based on assigned scripts and current neighbors.
            # Invariant: All assigned scripts are converted into executable tasks for the work pool.
            for (script, location) in self.device.scripts:
                tasks.append((script, location, neighbours, self.device))

            # Block Logic: If tasks are available, they are set to the work pool, and the thread waits
            # for their completion.
            # Invariant: Scripts are processed if available, and the thread awaits their execution.
            if tasks != []:
                # Assigns the collected tasks to the device's work pool for execution.
                self.device.work_pool.set_tasks(tasks)

                # Waits until all tasks in the work pool have finished execution.
                self.tasks_finish.wait()

                # Clears the tasks_finish event, preparing for the next set of tasks.
                self.tasks_finish.clear()

            # Clears the script_received event, indicating readiness for new script assignments.
            self.device.script_received.clear()
