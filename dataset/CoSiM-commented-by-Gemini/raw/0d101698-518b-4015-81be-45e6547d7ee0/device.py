# -*- coding: utf-8 -*-
"""
This module implements a distributed device simulation using a centralized
task scheduler architecture.

In this model, a single TaskScheduler object is shared across all devices.
This scheduler manages a global pool of worker threads and a central task queue.
Device threads act as producers, adding tasks to the queue, while the global
workers act as consumers, executing the tasks.

Classes:
    Device: A node that submits tasks to the central scheduler.
    DeviceThread: The main control loop for a device.
    Task: A data object representing a single unit of work.
    TaskScheduler: The central scheduler managing a global queue and worker pool.
    Worker: A global worker thread that consumes tasks from the scheduler.
"""

from threading import Event, Thread, Lock
# Assumes ReusableBarrierSem is defined in a 'barrier' module.
from barrier import ReusableBarrierSem
# The following are defined at the end of this file, but are used here.
# from taskscheduler import TaskScheduler
# from task import Task


def add_lock_for_location(lock_per_location, location):
    """Helper to add a new lock for a location."""
    lock_per_location.append((location, Lock()))

def get_lock_for_location(lock_per_location, location):
    """Helper to find the lock for a given location."""
    for (loc, lock) in lock_per_location:
        if loc == location:
            return lock
    return None


class Device(object):
    """Represents a device that offloads its work to a central scheduler."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.script_received = Event() # Not used in this implementation.
        self.timepoint_done = Event()
        # The barrier and scheduler are shared and will be assigned during setup.
        self.barrier = None
        self.taskscheduler = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources, led by device 0.
        
        The leader (device 0) creates a single TaskScheduler and a single
        ReusableBarrier and distributes them to all other devices.
        """
        if self.device_id == 0:
            # Create a list of (location, Lock) tuples.
            lock_per_location = []
            all_locations = set(loc for dev in devices for loc in dev.sensor_data)
            for loc in all_locations:
                add_lock_for_location(lock_per_location, loc)

            # Create the single shared barrier and scheduler.
            self.barrier = ReusableBarrierSem(len(devices))
            self.taskscheduler = TaskScheduler(lock_per_location)

            # Distribute the shared objects.
            for device in devices:
                device.taskscheduler = self.taskscheduler
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to be processed in the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """The main control loop for a device, which submits tasks to the scheduler."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation loop.
        
        Synchronization Logic:
        1. Waits for the supervisor to assign all scripts.
        2. Submits all its assigned scripts as Task objects to the central scheduler.
        3. Waits at a global barrier. This signals that all devices have submitted
           their work for the timepoint. The actual work is done asynchronously.
        """
        # Busy-wait until setup is complete and the barrier is assigned.
        while self.device.barrier is None:
            pass

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # The leader device signals the scheduler to shut down.
                if self.device.device_id == 0:
                    self.device.taskscheduler.finish = True
                    self.device.taskscheduler.wait_workers()
                break

            # 1. Wait for scripts to be assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # 2. Add all assigned work to the central task scheduler's queue.
            for (script, location) in self.device.scripts:
                new_task = Task(self.device, script, location, neighbours)
                self.device.taskscheduler.add_task(new_task)
            self.device.scripts = []

            # 3. Synchronize with other devices to end the submission phase.
            self.device.barrier.wait()


class Task(object):
    """A data class that encapsulates a single unit of work."""

    def __init__(self, device, script, location, neighbours):
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def execute(self):
        """Contains the core logic: gather data, run script, update data."""
        script_data = []
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

# --- The classes below would typically be in their own modules ---

class TaskScheduler(object):
    """
    A central scheduler that manages a global pool of worker threads and a
    single queue of tasks.
    """
    def __init__(self, lock_per_location):
        self.nr_threads = 16
        self.lock_per_location = lock_per_location
        self.workpool = []
        self.workpool_lock = Lock()
        self.workers_list = []
        self.finish = False
        self.start_workers()

    def add_task(self, new_task):
        """Atomically adds a task to the global work pool."""
        with self.workpool_lock:
            self.workpool.append(new_task)

    def get_task(self):
        """Atomically retrieves a task from the global work pool."""
        with self.workpool_lock:
            return self.workpool.pop() if self.workpool else None

    def start_workers(self):
        """Creates and starts the global pool of worker threads."""
        self.workers_list = [Worker(self) for _ in range(self.nr_threads)]
        for worker in self.workers_list:
            worker.start()

    def wait_workers(self):
        """Waits for all worker threads to complete."""
        for worker in self.workers_list:
            worker.join()

    def get_lock_per_location(self, location):
        """Finds the lock associated with a given location."""
        for (loc, lock) in self.lock_per_location:
            if loc == location:
                return lock
        return None


class Worker(Thread):
    """A global worker thread that consumes and executes tasks."""
    def __init__(self, taskscheduler):
        Thread.__init__(self)
        self.taskscheduler = taskscheduler

    def run(self):
        """Continuously fetches and executes tasks from the scheduler."""
        while not self.taskscheduler.finish:
            task = self.taskscheduler.get_task()
            if task is None:
                continue
            
            # Acquire the specific lock for the task's location before execution.
            location_lock = self.taskscheduler.get_lock_per_location(task.location)
            if location_lock:
                with location_lock:
                    task.execute()