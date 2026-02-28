"""
Models a distributed device network using a thread pool architecture.

Each `Device` has an orchestrator thread (`DeviceThread`) which, for each
timepoint, submits a batch of tasks to a `WorkPool`. The `WorkPool` manages a
fixed set of `Worker` threads that execute the tasks. The system uses a global
barrier to synchronize devices between timepoints.
"""

from threading import Event, Thread, Lock
# Assumes a 'barrier' module with a ReusableBarrierCond class exists.
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device node, holding state and references to the
    main orchestrator thread and its work pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = None
        self.work_pool = None

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources and creates the main threads for all devices.

        The master device (ID 0) creates a global barrier and a list of shared
        locks, then distributes them. It also instantiates and starts the
        `WorkPool` and `DeviceThread` for every device in the network.
        """
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            lock_locations = []

            # Creates a list of locks. The logic does not map locks to specific
            # locations, but rather creates a number of locks based on the total
            # number of sensor entries across all devices.
            for device in devices:
                for _ in xrange(len(device.sensor_data)):
                    lock_locations.append(Lock())

            # For each device, create its work pool and main thread, then start it.
            for device in devices:
                tasks_finish = Event()
                device.work_pool = WorkPool(tasks_finish, lock_locations)
                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to this device's to-do list for the timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # `None` script signals that all work for the timepoint has been assigned.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data from a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class WorkPool(object):
    """
    Manages a pool of worker threads and a list of tasks for a device.
    """

    def __init__(self, tasks_finish, lock_locations):
        """Initializes and starts a fixed pool of worker threads."""
        self.workers = []
        self.tasks = []
        self.current_task_index = 0
        # This lock serializes access to the task list.
        self.lock_get_task = Lock()
        # Event to signal workers that there is work to do.
        self.work_to_do = Event()
        # Event to signal the orchestrator that all tasks have been *distributed*.
        self.tasks_finish = tasks_finish
        self.lock_locations = lock_locations
        self.max_num_workers = 8

        for i in xrange(self.max_num_workers):
            self.workers.append(Worker(self, self.lock_get_task, \
                    self.work_to_do, self.lock_locations))
            self.workers[i].start()

    def set_tasks(self, tasks):
        """
        Receives a new batch of tasks and signals the workers to start.
        """
        self.tasks = tasks
        self.current_task_index = 0
        self.work_to_do.set()

    def get_task(self):
        """
        Provides the next available task to a worker thread.

        Returns the next task, or `None` if the task list is exhausted. It signals
        `tasks_finish` when the last task is handed out, not when it's completed.
        """
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index]
            self.current_task_index += 1

            # If the last task was just handed out, signal completion of distribution.
            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear()
                self.tasks_finish.set()
            return task
        else:
            return None

    def close(self):
        """Shuts down the work pool by terminating and joining all workers."""
        self.tasks = []
        self.current_task_index = len(self.tasks)
        # Setting this event unblocks any waiting workers so they can receive
        # a `None` task and terminate.
        self.work_to_do.set()
        for worker in self.workers:
            worker.join()

class Worker(Thread):
    """A worker thread that executes tasks provided by a WorkPool."""

    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        Thread.__init__(self)
        self.work_pool = work_pool
        self.lock_get_task = lock_get_task
        self.work_to_do = work_to_do
        self.lock_locations = lock_locations

    def run(self):
        """Main loop: waits for work, acquires a task, and executes it."""
        while True:
            # This lock serializes task acquisition across all workers.
            self.lock_get_task.acquire()
            # Waits until the work pool signals that tasks are ready.
            self.work_to_do.wait()
            task = self.work_pool.get_task()
            self.lock_get_task.release()

            # A `None` task is the termination signal.
            if task is None:
                break

            script, location, neighbours, self_device = task

            # Acquire the specific lock for the data location being accessed.
            self.lock_locations[location].acquire()
            try:
                # --- Data Gathering and Execution ---
                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self_device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self_device.set_data(location, result)
            finally:
                self.lock_locations[location].release()

class DeviceThread(Thread):
    """
    The main orchestrator thread for a device. Manages the overall workflow
    for each timepoint.
    """
    def __init__(self, device, barrier, tasks_finish):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.tasks_finish = tasks_finish

    def run(self):
        """Main loop for managing timepoints."""
        while True:
            # Global sync: Wait for all devices to be ready for the new timepoint.
            self.barrier.wait()

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                self.device.work_pool.close()
                break # Shutdown signal

            # Wait for the supervisor to assign all scripts for this timepoint.
            self.device.script_received.wait()

            tasks = []
            for (script, location) in self.device.scripts:
                tasks.append((script, location, neighbours, self.device))

            if tasks:
                # Dispatch the batch of tasks to the work pool.
                self.device.work_pool.set_tasks(tasks)

                # Wait for the signal that all tasks have been *distributed* to workers.
                # This does NOT wait for the tasks to be completed.
                self.tasks_finish.wait()
                self.tasks_finish.clear()

            self.device.script_received.clear()
