from threading import Event, Thread, Lock
# This script depends on an external definition for ReusableBarrierSem.
from barrier import ReusableBarrierSem
# The following classes are defined later in this same file.
from taskscheduler import TaskScheduler
from task import Task


def add_lock_for_location(lock_per_location, location):
    """
    Helper function to add a new lock for a given location to the list.
    
    Args:
        lock_per_location (list): A list of (location, Lock) tuples.
        location (any): The location to associate with the new lock.
    """
    lock_per_location.append((location, Lock()))

def get_lock_for_location(lock_per_location, location):
    """
    Helper function to retrieve the lock for a given location.

    Args:
        lock_per_location (list): A list of (location, Lock) tuples.
        location (any): The location whose lock is to be retrieved.

    Returns:
        Lock or None: The lock object if found, otherwise None.
    """
    for (loc, lock) in lock_per_location:
        if loc == location:
            return lock
    return None


class Device(object):
    """
    Represents a device that receives scripts and offloads their execution
    as tasks to a central TaskScheduler.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and its main control thread.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The local data store for the device.
            supervisor (object): The supervisor managing the device network.
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

    def __str__(self):
        return "Device %d" % self.device_id

    def share_barrier(self, barrier):
        """Assigns the shared ReusableBarrier to this device."""
        self.barrier = barrier

    def share_taskscheduler(self, taskscheduler):
        """Assigns the shared TaskScheduler to this device."""
        self.taskscheduler = taskscheduler

    def setup_devices(self, devices):
        """

        Initializes shared resources for the entire device network.
        
        Device 0 acts as a primary, creating a shared barrier and a task
        scheduler (with fine-grained locks for each data location). It then
        distributes these shared objects to all other devices.
        """
        if self.device_id == 0:
            lock_per_location = []
            # Discover all unique data locations and create a lock for each.
            for device in devices:
                for location in device.sensor_data:
                    if get_lock_for_location(lock_per_location, location) is None:
                        add_lock_for_location(lock_per_location, location)

            # Create shared synchronization and scheduling objects.
            self.barrier = ReusableBarrierSem(len(devices))
            self.taskscheduler = TaskScheduler(lock_per_location)

            # Distribute shared objects to all other devices.
            for device in devices:
                device.share_taskscheduler(self.taskscheduler)
                device.share_barrier(self.barrier)

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals the end of a timepoint.

        Args:
            script (object): The script to run, or None.
            location (any): The data location for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals all scripts for the timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device. Its role is to package scripts
    into tasks and submit them to the central TaskScheduler.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for the device thread.
        """
        # Busy-wait until the shared objects are initialized by device 0.
        while self.device.barrier is None:
            pass

        while True:
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # Supervisor signals shutdown.
                if self.device.device_id == 0:
                    # Device 0 is responsible for shutting down the scheduler.
                    self.device.taskscheduler.finish = True
                    self.device.taskscheduler.wait_workers()
                break

            # Wait for the supervisor to signal that all scripts for this
            # timepoint have been assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Create Task objects for all assigned scripts and add them to the scheduler's work pool.
            for (script, location) in self.device.scripts:
                new_task = Task(self.device, script, location, neighbours)
                self.device.taskscheduler.add_task(new_task)

            # After submitting all tasks, wait at the barrier for all other
            # devices to also finish submitting their tasks.
            self.device.barrier.wait()


class Task(object):
    """
    A data class representing a single unit of work to be executed by the TaskScheduler.
    """
    def __init__(self, device, script, location, neighbours):
        """
        Initializes a task.
        
        Args:
            device (Device): The device that originated the task.
            script (object): The script object to be executed.
            location (any): The data location for the script.
            neighbours (list): List of neighboring devices.
        """
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def execute(self):
        """
        Executes the script. This involves gathering data from the originating
        device and its neighbors, running the script, and broadcasting the result.
        """
        script_data = []
        # Gather data from neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
		# Gather data from the local device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Run the script.
            result = self.script.run(script_data)
            # Broadcast the result by overwriting data on all relevant devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)


from threading import Thread, Lock

class TaskScheduler(object):
    """
    A central, thread-safe scheduler that manages a pool of worker threads
    to execute tasks concurrently.
    """
    def __init__(self, lock_per_location):
        """
        Initializes the TaskScheduler and starts its worker pool.

        Args:
            lock_per_location (list): A list of (location, Lock) tuples for
                                      fine-grained locking.
        """
        self.nr_threads = 16
        self.lock_per_location = lock_per_location
        self.workpool = []
        self.workpool_lock = Lock()
        self.workers_list = []
        self.finish = False
        self.start_workers()

    def add_task(self, new_task):
        """Atomically adds a new task to the work pool."""
        with self.workpool_lock:
            self.workpool.append(new_task)

    def get_task(self):
        """Atomically retrieves a task from the work pool."""
        with self.workpool_lock:
            return self.workpool.pop() if self.workpool else None

    def start_workers(self):
        """Creates and starts the fixed-size pool of worker threads."""
        for _ in range(self.nr_threads):
            worker = Worker(self)
            self.workers_list.append(worker)
            worker.start()

    def wait_workers(self):
        """Waits for all worker threads to complete."""
        for worker in self.workers_list:
            worker.join()

    def get_lock_per_location(self, location):
        """Retrieves the specific lock for a given data location."""
        for (loc, lock) in self.lock_per_location:
            if loc == location:
                return lock
        return None


class Worker(Thread):
    """
    A worker thread that executes tasks from the TaskScheduler's work pool.
    """
    def __init__(self, taskscheduler):
        Thread.__init__(self)
        self.taskscheduler = taskscheduler

    def run(self):
        """
        Main loop for the worker. It continuously fetches and executes tasks
        until the scheduler is finished.
        """
        while not self.taskscheduler.finish:
            while True:
                # Fetch a task from the shared pool.
                task = self.taskscheduler.get_task()
                if task is None:
                    # No more tasks in the pool for now.
                    break
                
                # Acquire the lock for the task's specific data location before execution.
                with self.taskscheduler.get_lock_per_location(task.location):
                    task.execute()
