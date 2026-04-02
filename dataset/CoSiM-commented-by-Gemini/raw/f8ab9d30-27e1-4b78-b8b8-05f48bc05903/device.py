"""
This module presents a fifth variation of the distributed device simulation.

This architecture elects a single "global" device (the one with the lowest ID)
to be the owner of all shared synchronization primitives, including a global
reusable barrier and a dictionary of location-based locks. Each device in the
simulation maintains its own independent thread pool to process tasks.
Synchronization occurs at two levels: workers within a device are synchronized
when the device's task queue is exhausted, and all devices are synchronized
globally at a barrier before proceeding to the next time-step.
"""

from threading import Event, Thread, Lock
from operator import attrgetter
from barrier import ReusableBarrierCond
from pool import ThreadPool

class Device(object):
    """
    Represents a device node in the simulation, managing a local thread pool
    and interacting with globally shared synchronization objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()
        self.scripts = []

        self.thread = DeviceThread(self)
        self.other_devices = []

        # A reference to the "global" device that owns shared resources.
        self.gdevice = None
        self.gid = None

        # A reference to the globally shared dictionary of locks.
        self.glocks = {}

        # A reference to the globally shared barrier.
        self.barrier = None

        # The device's local thread pool for executing script tasks.
        self.threadpool = None

        # The number of worker threads in the local pool.
        self.nthreads = 8

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the device and its connection to shared resources.

        This method elects a global device (gdevice) based on the lowest device_id.
        The gdevice is responsible for creating the shared barrier and location locks.
        All devices then receive a reference to these shared objects.
        """
        self.other_devices = devices

        # Elect the device with the minimum ID as the global device.
        self.gdevice = min(devices, key=attrgetter('device_id'))
        self.gid = self.gdevice.device_id

        # The global device creates the shared resources.
        if self.device_id == self.gid:
            # Collect all unique locations from all devices.
            list_loc = []
            for dev in self.other_devices:
                for key, _ in dev.sensor_data.iteritems():
                    list_loc.append(key)
            list_nodup = list(set(list_loc))

            # Create a lock for each unique location.
            locks = {}
            for loc in list_nodup:
                locks[loc] = Lock()
            self.glocks = locks
            
            # Create a shared barrier for all devices.
            self.barrier = ReusableBarrierCond(len(self.other_devices))

        # Each device initializes its own thread pool.
        self.threadpool = ThreadPool(self.nthreads)

        # Start the main control thread for this device.
        self.thread.start()


    def assign_script(self, script, location):
        """
        Assigns a script to the device. A None script signals the end of
        assignments for the current timepoint.
        """

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the main device thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a single device. It orchestrates the execution
    of scripts for each timepoint using a local thread pool.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:

            # Get the list of neighboring devices for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()


            if neighbours is None:
                # A None value for neighbours signals the end of the simulation.
                self.device.threadpool.end()
                break

            # Wait for the signal indicating all scripts for this timepoint have been assigned.
            self.device.script_received.wait()

            # Add all assigned scripts as tasks to the local thread pool.
            for (script, location) in self.device.scripts:
                self.device.threadpool.add_task((self.device, script, location, neighbours))
            # Wait for all tasks in the local pool to be completed.
            self.device.threadpool.finish_tasks()

            # Reset the event for the next timepoint.
            self.device.script_received.clear()

            # Wait at the global barrier for all other devices to complete their timepoints.
            self.device.gdevice.barrier.wait()


from threading import Lock, Thread
from Queue import Queue

class WorkerThread(Thread):
    """
    A worker thread from a thread pool that executes computational tasks.
    """

    def __init__(self, threadpool):
        """Initializes and starts the worker thread."""
        Thread.__init__(self, name="worker")
        self.threadpool = threadpool
        self.start()

    def run(self):
        """Continuously fetches and executes tasks from the pool's queue."""
        while True:
            # Check for the shutdown signal.
            if self.threadpool.stop:
                break

            current_task = None

            # Atomically check for and retrieve a task from the queue.
            # Note: This is an unconventional polling approach. A simple blocking
            # `get()` would be more efficient.
            with self.threadpool.task_lock:
                if self.threadpool.tasks.qsize() > 0:
                    current_task = self.threadpool.tasks.get_nowait()

            # If a task was retrieved, execute it.
            if current_task is not None:
                (device, script, location, neighbours) = current_task

                # Acquire the global lock for this specific location to ensure exclusive access.
                with device.gdevice.glocks[location]:
                    script_data = []
                    
                    # Gather data from all neighboring devices.
                    for dev in neighbours:
                        data = dev.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    # Gather data from the local device.
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    # Execute the script and disseminate the results.
                    if script_data != []:
                        
                        result = script.run(script_data)
                        
                        for dev in neighbours:
                            dev.set_data(location, result)
                        
                        device.set_data(location, result)

                # Signal to the queue that the task is complete.
                self.threadpool.tasks.task_done()


class ThreadPool(object):
    """
    A simple thread pool implementation to manage worker threads.
    """

    def __init__(self, size):
        """Initializes the thread pool and starts the worker threads."""
        self.size = size
        self.tasks = Queue()
        self.workers = []
        self.task_lock = Lock()
        self.stop = False

        for _ in xrange(self.size):
            self.workers.append(WorkerThread(self))

    def add_task(self, task):
        """Adds a task to the queue for a worker to execute."""
        with self.task_lock:
            self.tasks.put(task)

    def finish_tasks(self):
        """Blocks until all tasks in the queue have been completed."""
        self.tasks.join()

    def end(self):
        """Shuts down the thread pool and all its worker threads."""
        self.tasks.join()
        self.stop = True
        for thread in self.workers:
            thread.join()
