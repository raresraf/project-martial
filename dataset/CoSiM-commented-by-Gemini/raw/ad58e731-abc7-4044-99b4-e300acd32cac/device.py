"""
This module defines a multi-threaded simulation framework for a network of devices.

It consists of four main classes:
- Device: Represents a single entity in the network, holding sensor data and
  assigned scripts.
- DeviceThread: The main control thread for a single Device, orchestrating its
  lifecycle and synchronization with other devices.
- WorkPool: Manages a pool of worker threads to execute tasks concurrently.
- Worker: A thread that executes a single script on a device's data and its
  neighbors' data.

The system uses threading primitives like Events, Locks, and a custom
ReusableBarrierCond for synchronization between devices and workers.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device has an ID, local sensor data, and can be assigned scripts to
    process this data in coordination with its neighbors.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor  # Manages the network topology.
        self.script_received = Event()  # Signals when all scripts for a timepoint are assigned.
        self.scripts = []  # A list of (script, location) tuples to be executed.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint.
        self.thread = None  # The main thread for this device.
        self.work_pool = None  # The thread pool for executing script tasks.

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and starts the simulation for a list of devices.

        This method should only be called on one device (e.g., device 0) to
        set up the shared synchronization objects for all devices.
        """
        # This setup is centralized and should only be run once.
        if self.device_id == 0:
            # A barrier to synchronize all device threads at the start of each timepoint.
            barrier = ReusableBarrierCond(len(devices))
            lock_locations = []

            # Create a unique lock for each potential data location across all devices.
            for device in devices:
                for _ in xrange(len(device.sensor_data)):
                    lock_locations.append(Lock())

            # Initialize and start a thread and work pool for each device.
            for device in devices:
                tasks_finish = Event()
                device.work_pool = WorkPool(tasks_finish, lock_locations)

                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location.

        If the script is None, it signals that all scripts for the current
        timepoint have been assigned.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # All scripts for this cycle have been received; unblock the device thread.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data from a specified location."""
        return self.sensor_data[location] if location in self.sensor_data \
                else None

    def set_data(self, location, data):
        """Updates sensor data at a specified location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread."""
        self.thread.join()


class WorkPool(object):
    """
    Manages a pool of worker threads to execute computation tasks.

    This class distributes a set of tasks among a fixed number of worker threads.
    """

    def __init__(self, tasks_finish, lock_locations):
        """Initializes the WorkPool with shared synchronization objects."""
        self.workers = []
        self.tasks = []
        self.current_task_index = 0
        self.lock_get_task = Lock()  # Protects access to the task queue.
        self.work_to_do = Event()  # Signals workers that new tasks are available.
        self.tasks_finish = tasks_finish  # Signals the device thread that all tasks are done.
        self.lock_locations = lock_locations  # Locks for exclusive access to data locations.
        self.max_num_workers = 8  # The number of concurrent worker threads.

        for i in xrange(self.max_num_workers):
            self.workers.append(Worker(self, self.lock_get_task, \
                    self.work_to_do, self.lock_locations))
            self.workers[i].start()

    def set_tasks(self, tasks):
        """
        Assigns a new list of tasks to the pool and wakes up the workers.
        """
        self.tasks = tasks
        self.current_task_index = 0

        self.work_to_do.set()

    def get_task(self):
        """
        Atomically retrieves the next available task from the queue.

        Returns None if no tasks are left.
        """
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index]
            self.current_task_index = self.current_task_index + 1

            # If this was the last task, signal completion.
            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear()
                self.tasks_finish.set()

            return task
        else:
            return None

    def close(self):
        """Shuts down all worker threads in the pool."""
        self.tasks = []
        self.current_task_index = len(self.tasks)

        # Wake up all workers so they can check the empty task list and exit.
        self.work_to_do.set()

        for worker in self.workers:
            worker.join()


class Worker(Thread):
    """
    A worker thread that executes tasks from a WorkPool.

    Each worker fetches a task, acquires the necessary locks, executes the
    script, and updates the data.
    """

    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        """Initializes the Worker."""
        Thread.__init__(self)
        self.work_pool = work_pool
        self.lock_get_task = lock_get_task
        self.work_to_do = work_to_do
        self.lock_locations = lock_locations

    def run(self):
        """The main loop for the worker thread."""
        while True:
            # Acquire lock to ensure atomic task retrieval.
            self.lock_get_task.acquire()

            # Wait until there is work to do.
            self.work_to_do.wait()

            task = self.work_pool.get_task()

            # Release lock immediately after getting a task so other workers can proceed.
            self.lock_get_task.release()

            # If get_task() returns None, it's a signal to shut down.
            if task is None:
                break

            # Unpack task details.
            script, location, neighbours, self_device = task

            # Acquire a lock for the specific data location to prevent race conditions.
            self.lock_locations[location].acquire()

            script_data = []

            # Gather data from neighboring devices and the local device.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self_device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Execute the script with the gathered data.
                result = script.run(script_data)

                # Broadcast the result back to all involved devices.
                for device in neighbours:
                    device.set_data(location, result)
                
                self_device.set_data(location, result)

            # Release the lock for the data location.
            self.lock_locations[location].release()


class DeviceThread(Thread):
    """
    The main control thread for a single Device instance.

    This thread synchronizes with other devices at each timepoint, processes
    assigned scripts, and waits for the work pool to complete its tasks.
    """

    def __init__(self, device, barrier, tasks_finish):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier  # Shared barrier for all device threads.
        self.tasks_finish = tasks_finish  # Event to signal task completion.

    def run(self):
        """The main lifecycle loop of the device."""
        while True:
            # Synchronize all device threads at the start of a new timepoint.
            self.barrier.wait()

            # Get the list of neighbors for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()

            # A None value for neighbours is the signal to terminate.
            if neighbours is None:
                self.device.work_pool.close()
                break

            # Wait until the supervisor has assigned all scripts for this timepoint.
            self.device.script_received.wait()

            # Prepare tasks for the work pool from the assigned scripts.
            tasks = []
            for (script, location) in self.device.scripts:
                tasks.append((script, location, neighbours, self.device))

            if tasks:
                # Assign tasks to the work pool and wait for them to finish.
                self.device.work_pool.set_tasks(tasks)
                self.tasks_finish.wait()
                self.tasks_finish.clear()

            # Clear the script list and events in preparation for the next timepoint.
            self.device.script_received.clear()