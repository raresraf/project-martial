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

"""
Defines a distributed device simulation using a thread pool architecture.

This file contains the components for a simulation framework where devices
execute tasks in parallel. The architecture consists of:
- `ThreadPool` and `Worker`: A classic worker-pool implementation for task execution.
- `Device`: Represents a node in the network, holding data and state.
- `DeviceThread`: The main control loop for a device, which dispatches tasks
  to the `ThreadPool`.

Note: The file is named `ThreadPool.py` but contains the entire device
simulation logic. The implementation contains apparent race conditions in how
shared synchronization objects are initialized.
"""

from Queue import Queue
from threading import Thread, Event, Lock
from barrier import ReusableBarrierSem


class Worker(Thread):
    """A worker thread that executes tasks from a shared queue."""

    def __init__(self, tasks, device):
        """Initializes the worker.

        Args:
            tasks (Queue): The shared task queue.
            device (Device): The device this worker belongs to.
        """
        Thread.__init__(self)
        self.tasks = tasks
        self.device = device
        self.daemon = True
        self.start()

    def run(self):
        """The main execution loop for the worker thread.

        Continuously fetches tasks from the queue and executes them. A task
        is a tuple of (neighbours, script, location). A `None` value for
        `neighbours` is a sentinel to terminate the thread.
        """
        while True:
            # Get a task from the queue. A None task is a sentinel to exit.
            neighbours, script, location = self.tasks.get()

            # Sentinel check for thread shutdown.
            if neighbours is None:
                self.tasks.task_done()
                break

            # Acquire a lock for the specific data location to ensure exclusive access.
            with self.device.locations_locks[location]:
                self._script(neighbours, script, location)
            self.tasks.task_done()

    def _script(self, neighbours, script, location):
        """Executes the logic for a single script."""
        # Gather data from neighbors and self.
        script_data = []
        for neighbour in neighbours:
            data = neighbour.get_data(location)
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # Run the script and propagate the results.
        if script_data:
            result = script.run(script_data)
            for neighbour in neighbours:
                neighbour.set_data(location, result)
            self.device.set_data(location, result)


class ThreadPool(object):
    """Manages a pool of worker threads and a task queue."""

    def __init__(self, num_threads):
        """Initializes the thread pool."""
        self.tasks = Queue(num_threads)
        self.threads = []
        self.device = None

    def set_device(self, device, num_threads):
        """Links the pool to a device and creates the worker threads."""
        self.device = device
        for _ in range(num_threads):
            self.threads.append(Worker(self.tasks, self.device))

    def add_tasks(self, neighbours, location, script):
        """Adds a new task to the queue for the workers to process."""
        self.tasks.put((neighbours, location, script))

    def wait_completion(self):
        """Blocks until all tasks in the queue have been processed."""
        self.tasks.join()

    def end_threads(self):
        """Shuts down all worker threads in the pool gracefully."""
        self.tasks.join()
        # Add a sentinel task for each thread to cause it to exit its loop.
        for _ in range(len(self.threads)):
            self.add_tasks(None, None, None)
        for thread in self.threads:
            thread.join()


class Device(object):
    """Represents a node in the distributed simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device."""
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
        """
        Initializes synchronization objects for the device group.

        Note: This implementation appears to have a race condition. Each device
        creates its own barrier and then assigns it to all other devices. The
        last device to execute this will determine the barrier for everyone.
        Similarly, `locations_locks` are overwritten, leading to unpredictable
        synchronization behavior.
        """
        barrier = ReusableBarrierSem(len(devices))
        self.barrier = barrier
        for dev in devices:
            dev.barrier = barrier
            dev.locations_locks = self.locations_locks

    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        Note: This method contains a bug. It creates a new `Lock()` for the
        given location, overwriting the shared lock that was intended to be
        set up in `setup_devices`. This defeats the purpose of having a shared
        lock for that location across devices.
        """
        if script is not None:
            self.scripts.append((script, location))
            # This line overwrites the shared lock, breaking synchronization.
            self.locations_locks[location] = Lock()
        else:
            self.script_received.set()

    def get_data(self, location):
        """Gets data from a specific sensor location. Not thread-safe by itself."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a specific sensor location. Not thread-safe by itself."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a device's lifecycle."""

    def __init__(self, device):
        """Initializes the DeviceThread and its associated ThreadPool."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(8)

    def run(self):
        """The main orchestration loop.

        In each time step, this loop waits for scripts to be assigned, dispatches
        them to the thread pool, waits for their completion, and then synchronizes
        with all other devices at a global barrier.
        """
        self.thread_pool.set_device(self.device, 8)

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown.
                break

            # Wait until all scripts for the time step are assigned.
            self.device.script_received.wait()

            # Add all assigned scripts to the thread pool's task queue.
            for (script, location) in self.device.scripts:
                self.thread_pool.add_tasks(neighbours, script, location)

            self.device.script_received.clear()
            # Wait for the thread pool to process all tasks for this time step.
            self.thread_pool.wait_completion()
            # Synchronize with all other devices before starting the next time step.
            self.device.barrier.wait()

        # Cleanly shut down the thread pool when the simulation ends.
        self.thread_pool.end_threads()
