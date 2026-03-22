"""
This module implements a distributed device simulation with a decentralized
architecture where each device manages its own thread pool.

Key architectural features:
- A "master" device (device 0) creates and distributes a shared global barrier
  and a shared list of location-specific locks.
- Each device is initialized with its own `WorkPool` instance, making the
  computation handling decentralized.
- The main control threads (`DeviceThread`) are synchronized at the start of each
  time step by the global barrier.
- Coordination between a `DeviceThread` and its local `WorkPool` is handled
  by `Event` objects.
- Worker threads acquire tasks one by one, serialized by a lock, which can be
  a performance bottleneck.

Note: This script depends on a `barrier.py` module and uses Python 2 syntax.
"""


from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a device node which manages its own thread pool and control thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = None
        self.work_pool = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources using a master-worker pattern and starts threads.
        
        Device 0 acts as the master, creating a shared barrier and a shared list
        of locks. It also creates a unique `WorkPool` for each device.
        """
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            lock_locations = []

            # Create a global list of locks, one for each possible sensor location.
            for device in devices:
                for _ in xrange(len(device.sensor_data)):
                    lock_locations.append(Lock())

            # For each device, create its own WorkPool and control thread.
            for device in devices:
                tasks_finish = Event()
                device.work_pool = WorkPool(tasks_finish, lock_locations)

                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        """Assigns a script for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Non-thread-safe method to get data."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Non-thread-safe method to set data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class WorkPool(object):
    """
    A per-device thread pool that manages a batch of tasks for one time step.
    """
    def __init__(self, tasks_finish, lock_locations):
        self.workers = []
        self.tasks = []
        self.current_task_index = 0
        self.lock_get_task = Lock()
        self.work_to_do = Event()
        self.tasks_finish = tasks_finish
        self.lock_locations = lock_locations
        self.max_num_workers = 8

        for i in xrange(self.max_num_workers):
            self.workers.append(Worker(self, self.lock_get_task, \
                    self.work_to_do, self.lock_locations))
            self.workers[i].start()

    def set_tasks(self, tasks):
        """Provides a new batch of tasks to the pool and wakes up workers."""
        self.tasks = tasks
        self.current_task_index = 0
        self.work_to_do.set()

    def get_task(self):
        """
        Called by workers to get the next available task from the current batch.
        
        When the last task is dispensed, it signals the `tasks_finish` event to
        notify the parent DeviceThread.
        :return: A task tuple or None if no tasks are left.
        """
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index]
            self.current_task_index += 1

            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear()
                self.tasks_finish.set() # Signal completion to DeviceThread.

            return task
        return None

    def close(self):
        """Shuts down the work pool by signaling and joining all workers."""
        self.tasks = []
        self.current_task_index = len(self.tasks)
        self.work_to_do.set() # Wake up workers so they can exit.
        for worker in self.workers:
            worker.join()

class Worker(Thread):
    """
    A worker thread that executes tasks provided by its WorkPool.
    """
    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        Thread.__init__(self)
        self.work_pool = work_pool
        self.lock_get_task = lock_get_task
        self.work_to_do = work_to_do
        self.lock_locations = lock_locations

    def run(self):
        """Main loop to acquire and execute tasks."""
        while True:
            # This serialized task acquisition can be a bottleneck, as workers
            # line up to get tasks one by one.
            with self.lock_get_task:
                self.work_to_do.wait() # Wait for a batch of work to be ready.
                task = self.work_pool.get_task()

            if task is None:
                # If get_task returns None and work_to_do is set, it's a shutdown signal.
                if self.work_pool.current_task_index == len(self.work_pool.tasks):
                    break

            script, location, neighbours, self_device = task

            # Acquire the global lock for the specific location.
            with self.lock_locations[location]:
                script_data = []

                # Gather data, execute script, and broadcast results.
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

class DeviceThread(Thread):
    """
    The main control thread for a device. It coordinates with the global barrier
    and its local WorkPool.
    """
    def __init__(self, device, barrier, tasks_finish):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.tasks_finish = tasks_finish

    def run(self):
        """The main execution loop, synchronized by a global barrier."""
        while True:
            # BARRIER: All DeviceThreads wait here for the time step to start.
            self.barrier.wait()

            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                self.device.work_pool.close()
                break

            # Wait for the supervisor to assign all scripts for this step.
            self.device.script_received.wait()

            tasks = []
            for (script, location) in self.device.scripts:
                tasks.append((script, location, neighbours, self.device))

            if tasks:
                # Submit tasks to the local work pool.
                self.device.work_pool.set_tasks(tasks)
                # Wait for the local work pool to finish all its tasks.
                self.tasks_finish.wait()
                self.tasks_finish.clear()
            
            self.device.script_received.clear()
