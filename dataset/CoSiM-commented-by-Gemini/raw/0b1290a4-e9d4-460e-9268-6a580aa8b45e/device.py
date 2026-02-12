# -*- coding: utf-8 -*-
"""
This module implements a distributed device simulation using a thread pool
architecture. It defines a `Device` that participates in synchronized,
timepoint-based data processing. Each device uses a `WorkPool` of persistent
`Worker` threads to execute assigned scripts, which is an efficient alternative
to a thread-per-task model.

Classes:
    Device: Represents a node in the network.
    WorkPool: Manages a pool of worker threads for a device.
    Worker: A reusable thread that executes tasks from the work pool.
    DeviceThread: The main control loop for a device, orchestrating timepoints.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device in the simulation.

    This device uses a WorkPool to manage concurrent execution of data
    processing scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (object): The central coordinator of the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.script_received = Event()
        self.thread = None      # The main device thread, created in setup.
        self.work_pool = None   # The thread pool for this device, created in setup.

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources and starts the main thread for each device.

        The device with ID 0 acts as a leader to create a shared barrier and a
        list of locks for all locations. It then instantiates a WorkPool and
        a DeviceThread for every device in the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # This setup must be run by a single device to ensure all devices
        # share the same synchronization objects.
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            
            # Create a flat list of locks. The 'location' for a script will be
            # used as an index into this list. This assumes locations are integers.
            num_locations = sum(len(d.sensor_data) for d in devices)
            lock_locations = [Lock() for _ in xrange(num_locations)]

            # For each device, create its own WorkPool and main thread.
            for device in devices:
                tasks_finish = Event() # Event to signal a device's WorkPool is done.
                device.work_pool = WorkPool(tasks_finish, lock_locations)
                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. If script is None, it signals that
        all scripts for the current timepoint have been assigned.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location, if it exists."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data for a given location, if it exists."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        if self.thread:
            self.thread.join()

class WorkPool(object):
    """
    A thread pool to manage a set of reusable Worker threads.
    
    This avoids the overhead of creating new threads for every task.
    """

    def __init__(self, tasks_finish, lock_locations):
        """
        Initializes the WorkPool.

        Args:
            tasks_finish (Event): An event to signal when all tasks are complete.
            lock_locations (list): A list of shared locks for data locations.
        """
        self.tasks = []
        self.current_task_index = 0
        self.lock_get_task = Lock()
        self.work_to_do = Event()
        self.tasks_finish = tasks_finish
        self.lock_locations = lock_locations
        self.max_num_workers = 8

        # Create and start a fixed number of worker threads.
        self.workers = [
            Worker(self, self.lock_get_task, self.work_to_do, self.lock_locations)
            for _ in xrange(self.max_num_workers)
        ]
        for worker in self.workers:
            worker.start()

    def set_tasks(self, tasks):
        """
        Provides a new set of tasks to the pool and wakes up the workers.
        """
        self.tasks = tasks
        self.current_task_index = 0
        self.work_to_do.set()

    def get_task(self):
        """
        Atomically retrieves the next task from the list for a worker.
        
        Returns:
            A task tuple, or None if no tasks are left or if shutting down.
        """
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index]
            self.current_task_index += 1

            # If the last task has just been taken, notify the main device thread.
            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear() # Put workers to sleep.
                self.tasks_finish.set() # Signal completion.
            return task
        return None

    def close(self):
        """Shuts down the work pool by stopping all worker threads."""
        self.tasks = []
        self.current_task_index = 0  # Ensure get_task returns None
        self.work_to_do.set() # Wake up workers to allow them to exit.
        for worker in self.workers:
            worker.join()

class Worker(Thread):
    """A worker thread that executes tasks from a WorkPool."""

    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        Thread.__init__(self)
        self.work_pool = work_pool
        self.lock_get_task = lock_get_task
        self.work_to_do = work_to_do
        self.lock_locations = lock_locations

    def run(self):
        """The main loop for a worker thread."""
        while True:
            # Wait until there is work to be done.
            self.work_to_do.wait()

            self.lock_get_task.acquire()
            task = self.work_pool.get_task()
            self.lock_get_task.release()

            if task is None: # Signal to terminate the worker.
                break

            script, location, neighbours, self_device = task

            # Acquire the lock for the specific location to ensure data consistency.
            self.lock_locations[location].acquire()
            try:
                # --- Data Gathering ---
                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self_device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # --- Execution and Dissemination ---
                if script_data:
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self_device.set_data(location, result)
            finally:
                # Always release the lock.
                self.lock_locations[location].release()


class DeviceThread(Thread):
    """The main control loop for a device, synchronized with other devices."""

    def __init__(self, device, barrier, tasks_finish):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.tasks_finish = tasks_finish

    def run(self):
        """
        The main simulation loop, advancing in discrete timepoints.
        
        Synchronization logic:
        1. All devices wait at a barrier, ensuring they start the timepoint together.
        2. Each device receives scripts and passes them to its own WorkPool.
        3. Each device waits for its WorkPool to finish all assigned tasks.
        4. The loop repeats, waiting at the barrier for the next timepoint.
        """
        while True:
            # --- First Synchronization Point ---
            # All devices wait here before starting the timepoint.
            self.barrier.wait()

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Supervisor signals shutdown.
                self.device.work_pool.close()
                break

            # Wait for the supervisor to assign all scripts for this timepoint.
            self.device.script_received.wait()

            tasks = [
                (script, location, neighbours, self.device)
                for (script, location) in self.device.scripts
            ]

            if tasks:
                # Delegate all tasks to the work pool.
                self.device.work_pool.set_tasks(tasks)
                # Wait until the work pool signals that all tasks are done.
                self.tasks_finish.wait()
                self.tasks_finish.clear()
            
            # Reset for the next timepoint.
            self.device.script_received.clear()
            self.device.scripts = []