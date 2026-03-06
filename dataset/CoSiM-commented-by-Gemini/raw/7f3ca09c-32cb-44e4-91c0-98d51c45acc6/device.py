"""
This module implements a distributed device simulation framework using a thread pool
pattern for task execution.

Each device is managed by a main `DeviceThread` that orchestrates the work for each
simulation timepoint. Instead of creating new threads for each task, it submits
tasks to a persistent `WorkPool` of `Worker` threads, which is a more efficient
model for handling numerous small tasks. Synchronization between devices is
managed by a reusable barrier, ensuring all devices complete a timepoint before
the next one begins.
"""

from threading import Event, Thread, Lock
# Assumes the existence of a 'barrier' module with a ReusableBarrierCond class.
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a device in the simulation, holding its state and configuration.

    This class acts as a container for the device's ID, sensor data, and its
    associated `DeviceThread` and `WorkPool`. It is responsible for receiving
    script assignments but delegates the execution and orchestration to its
    worker and controller threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (object): The supervisor managing the network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that all scripts for a timepoint have been received.
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()  # This Event appears unused in the current logic.
        self.thread = None
        self.work_pool = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and starts the simulation infrastructure.

        Executed by a single device (device_id 0), this method creates the shared
        synchronization barrier, location-specific locks, and a dedicated `WorkPool`
        and `DeviceThread` for every device in the simulation.

        Args:
            devices (list): A list of all Device objects.
        """
        # Pre-condition: This block should only be run once by a single master device.
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            lock_locations = []

            # Create a unique lock for each sensor data location across all devices.
            for device in devices:
                for _ in xrange(len(device.sensor_data)):
                    lock_locations.append(Lock())

            # For each device, create and start its own thread pool and controller thread.
            for device in devices:
                tasks_finish = Event()
                device.work_pool = WorkPool(tasks_finish, lock_locations)

                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current timepoint.

        A 'None' script acts as a sentinel value, signaling that all scripts for
        the current timepoint have been assigned.

        Args:
            script (object): The script to be executed.
            location (int): The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data 
                else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class WorkPool(object):
    """
    A thread pool for executing script tasks for a single device.

    Manages a fixed number of persistent `Worker` threads to avoid the overhead
    of thread creation/destruction for each task.
    """

    def __init__(self, tasks_finish, lock_locations):
        """
        Initializes the WorkPool.

        Args:
            tasks_finish (Event): An event to signal when all tasks in a batch are done.
            lock_locations (list): A list of shared, location-specific locks.
        """
        self.workers = []
        self.tasks = []
        self.current_task_index = 0
        self.lock_get_task = Lock()
        self.work_to_do = Event()
        self.tasks_finish = tasks_finish
        self.lock_locations = lock_locations
        self.max_num_workers = 8

        # Create and start the worker threads.
        for i in xrange(self.max_num_workers):
            self.workers.append(Worker(self, self.lock_get_task, 
                    self.work_to_do, self.lock_locations))
            self.workers[i].start()

    def set_tasks(self, tasks):
        """
        Assigns a new batch of tasks to the pool and wakes up worker threads.

        Args:
            tasks (list): A list of task tuples to be executed.
        """
        self.tasks = tasks
        self.current_task_index = 0
        # Signal to workers that there are new tasks available.
        self.work_to_do.set()

    def get_task(self):
        """
        Atomically retrieves the next available task from the queue.

        Returns:
            A task tuple if available, otherwise None.
        """
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index]
            self.current_task_index = self.current_task_index + 1

            # If the last task has just been taken, signal completion.
            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear()
                self.tasks_finish.set()

            return task
        else:
            return None

    def close(self):
        """Shuts down the work pool and all its worker threads."""
        self.tasks = []
        self.current_task_index = len(self.tasks) # Ensures get_task returns None

        # Wake up any waiting workers so they can see there are no tasks and exit.
        self.work_to_do.set()

        for worker in self.workers:
            worker.join()

class Worker(Thread):
    """
    A persistent worker thread that executes tasks from a WorkPool.
    """

    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        """Initializes the Worker."""
        Thread.__init__(self)
        self.work_pool = work_pool
        self.lock_get_task = lock_get_task
        self.work_to_do = work_to_do
        self.lock_locations = lock_locations

    def run(self):
        """
        The main loop of the worker thread.
        """
        # Invariant: The worker continuously waits for and processes tasks until
        # a 'None' task is received, signaling shutdown.
        while True:
            # The worker must acquire a lock before checking for work. This creates
            # a critical section around waiting for work and getting a task.
            self.lock_get_task.acquire()

            # Wait until the WorkPool signals that new tasks are available.
            self.work_to_do.wait()

            task = self.work_pool.get_task()

            # Release the lock immediately after getting a task, so other workers
            # can retrieve their tasks in parallel.
            self.lock_get_task.release()

            # A None task is the signal to terminate the worker thread.
            if task is None:
                break

            script, location, neighbours, self_device = task

            # Acquire the location-specific lock to ensure atomic read/write operations.
            self.lock_locations[location].acquire()

            script_data = []

            # Gather data from neighboring devices and the parent device.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = self_device.get_data(location)
            if data is not None:
                script_data.append(data)

            # If data was collected, run the script and broadcast the result.
            if script_data:
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                self_device.set_data(location, result)

            self.lock_locations[location].release()

class DeviceThread(Thread):
    """
    The main controller thread for a device, orchestrating its simulation cycle.
    """
    def __init__(self, device, barrier, tasks_finish):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread controls.
            barrier (ReusableBarrierCond): The shared barrier for all devices.
            tasks_finish (Event): Event to wait on for the WorkPool to finish.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.tasks_finish = tasks_finish

    def run(self):
        """
        The main simulation loop for the device.
        """
        while True:
            # Block until all devices in the simulation have reached this point.
            # This synchronizes the start of each timepoint.
            self.barrier.wait()

            neighbours = self.device.supervisor.get_neighbours()

            # If supervisor signals termination, shut down the work pool and exit.
            if neighbours is None:
                self.device.work_pool.close()
                break

            # Wait until the supervisor has assigned all scripts for this timepoint.
            self.device.script_received.wait()

            tasks = []
            for (script, location) in self.device.scripts:
                tasks.append((script, location, neighbours, self.device))

            if tasks:
                # Submit the batch of tasks to the work pool.
                self.device.work_pool.set_tasks(tasks)
                # Wait for the work pool to signal that all tasks are complete.
                self.tasks_finish.wait()
                self.tasks_finish.clear()

            # Clear the event and script list in preparation for the next timepoint.
            self.device.script_received.clear()
            self.device.scripts = []
