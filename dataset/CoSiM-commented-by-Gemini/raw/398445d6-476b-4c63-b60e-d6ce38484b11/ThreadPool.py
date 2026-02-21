"""
A thread pool-based simulation of distributed devices.

This module defines a simulation framework where each `Device` uses a dedicated
`ThreadPool` to execute computational scripts. The system is intended to operate
in synchronized time steps using a shared barrier.

NOTE: This file contains multiple class definitions and a local import,
`from ThreadPool import ThreadPool`, suggesting it was meant to be structured as
multiple files. It is documented here as a single module.

WARNING: This implementation contains critical bugs in the `Device` class related
to the setup and assignment of shared locks, which undermines the entire
synchronization scheme.
"""

from Queue import Queue
from threading import Thread

class Worker(Thread):
    """A worker thread that executes tasks from a queue.
    
    Each worker pulls a task, acquires the appropriate lock for the data
    location, executes the script, and releases the lock.
    """
    def __init__(self, tasks, device):
        """Initializes the worker and starts it as a daemon thread."""
        Thread.__init__(self)
        self.tasks = tasks
        self.device = device
        self.daemon = True
        self.start()

    def run(self):
        """The main loop for the worker thread."""
        while True:
            # Get a task from the queue. A None task is a sentinel to exit.
            neighbours, script, location = self.tasks.get()

            if neighbours is None:
                self.tasks.task_done()
                break
            
            # Use a 'with' statement for safe lock acquisition and release.
            with self.device.locations_locks[location]:
                self._script(neighbours, script, location)
            self.tasks.task_done()

    def _script(self, neighbours, script, location):
        """The core script execution logic."""
        # 1. Aggregate data from neighbors and self.
        script_data = []
        for neighbour in neighbours:
            data = neighbour.get_data(location)
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # 2. Run the script and disseminate the results.
        if script_data:
            result = script.run(script_data)
            for neighbour in neighbours:
                neighbour.set_data(location, result)
            self.device.set_data(location, result)


class ThreadPool(object):
    """A thread pool to manage and execute tasks in parallel."""
    def __init__(self, num_threads):
        """Initializes the task queue."""
        self.tasks = Queue(num_threads)
        self.threads = []
        self.device = None

    def set_device(self, device, num_threads):
        """Creates and starts the worker threads for a given device."""
        self.device = device
        for _ in range(num_threads):
            self.threads.append(Worker(self.tasks, self.device))

    def add_tasks(self, neighbours, location, script):
        """Adds a new task to the queue."""
        self.tasks.put((neighbours, location, script))

    def wait_completion(self):
        """Blocks until all tasks in the queue are processed."""
        self.tasks.join()

    def end_threads(self):
        """Gracefully shuts down all worker threads."""
        self.tasks.join()
        for _ in range(len(self.threads)):
            self.add_tasks(None, None, None) # Send sentinel values.
        for thread in self.threads:
            thread.join()


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
# This local import suggests the file was intended to be split.
from ThreadPool import ThreadPool

class Device(object):
    """Represents a device in the simulation.
    
    This class holds device state and manages a `DeviceThread` which in turn
    manages a `ThreadPool` for script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device, its locks, and its main thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.barrier = None
        self.locations_locks = {loc: Lock() for loc in sensor_data}
        self.devices = []

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources.
        
        WARNING: This implementation is flawed. Each device overwrites the
        `locations_locks` of all other devices with its own dictionary of
        locks. This creates a race condition where the last device to execute
        this loop determines the single set of locks used by all devices,
        instead of a master device creating and distributing one canonical set.
        """
        barrier = ReusableBarrierSem(len(devices))
        self.barrier = barrier
        for dev in devices:
            dev.barrier = barrier
            dev.locations_locks = self.locations_locks

    def assign_script(self, script, location):
        """Assigns a script to the device.

        WARNING: This method contains a critical bug. The line
        `self.locations_locks[location] = Lock()` overwrites the shared lock for
        a given location with a new, unshared lock. This breaks the intended
        synchronization between workers, as they will no longer be locking on
        the same object for that location.
        """
        if script is not None:
            self.scripts.append((script, location))
            # This line overwrites the shared lock, breaking synchronization.
            self.locations_locks[location] = Lock()
        else:
            self.script_received.set()

    def get_data(self, location):
        """Gets data from a location. (Locking is handled by the worker)."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a location. (Locking is handled by the worker)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main thread for a graceful shutdown."""
        self.thread.join()

class DeviceThread(Thread):
    """The main control loop for a device."""

    def __init__(self, device):
        """Initializes the thread and its dedicated thread pool."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(8)

    def run(self):
        """The main device lifecycle loop.
        
        Orchestrates work for a time step: waits for scripts, dispatches them
        to a thread pool, waits for completion, then syncs at a global barrier.
        """
        self.thread_pool.set_device(self.device, 8)

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Simulation exit condition.

            # Wait for the signal that all scripts for the step are assigned.
            self.device.script_received.wait()
            
            # Add all assigned scripts to the thread pool's task queue.
            for (script, location) in self.device.scripts:
                self.thread_pool.add_tasks(neighbours, script, location)

            self.device.script_received.clear()
            
            # Wait for all workers to finish their tasks for this step.
            self.thread_pool.wait_completion()
            
            # Synchronize with all other devices before the next step.
            self.device.barrier.wait()

        self.thread_pool.end_threads()
