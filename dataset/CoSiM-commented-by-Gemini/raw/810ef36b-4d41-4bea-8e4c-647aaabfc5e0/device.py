"""
This module implements a distributed device simulation using a robust thread pool
architecture. It is designed for clarity and safety in its concurrency model.

Each `Device` is managed by a single controller `DeviceThread`. This controller owns a
`ThreadPool` of long-lived `AuxiliaryDeviceThread` workers. For each simulation
timepoint, the controller places script-execution tasks into a shared `Queue`,
and the worker threads consume and execute these tasks in parallel.

Synchronization is handled cleanly through three mechanisms:
1. A thread-safe `Queue` for distributing tasks from the controller to the workers.
2. A shared dictionary of location-specific locks, ensuring that any single
   location is only processed by one worker at a time across the entire system.
3. An imported `ReusableBarrier` that synchronizes all controller threads between
   timepoints, ensuring no device starts the next step until all have finished
   the current one.
"""

import multiprocessing
from threading import Event, Thread, Lock
from Queue import Queue
# The following modules are assumed to be in the same directory.
from threadpool import ThreadPool
from reusablebarrier import ReusableBarrier


class Device(object):
    """
    Represents a single device in the network, holding its own data and
    managing its controller thread.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device and starts its controller thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.time_point_done = Event()
        self.nr_cpu = multiprocessing.cpu_count()
        self.thread = DeviceThread(self, self.nr_cpu)

        # Shared resources, to be populated by setup_devices.
        self.locations_lock_set = None
        self.barrier = None

        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (barrier, location locks)
        across all devices. Must be called by a single "master" device (device_id 0).
        """
        if self.device_id == 0:
            my_barrier = ReusableBarrier(len(devices))
            locations_lock_set = {}

            # Discover all unique locations and create a lock for each.
            for dev in devices:
                for location in dev.sensor_data:
                    if location not in locations_lock_set:
                        locations_lock_set[location] = Lock()

            # Distribute the shared barrier and lock set to all devices.
            for dev in devices:
                dev.barrier = my_barrier
                dev.locations_lock_set = locations_lock_set

    def assign_script(self, script, location):
        """Adds a script to the device's list for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A 'None' script is a sentinel from the supervisor indicating
            # all scripts have been assigned.
            self.time_point_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data. Note: Not thread-safe by itself. Relies on
        the calling worker thread to hold the appropriate location lock.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Sets sensor data. Note: Not thread-safe by itself. Relies on the
        calling worker thread to hold the appropriate location lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's controller thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main controller thread for a single Device. It manages a ThreadPool
    and orchestrates the device's participation in each simulation timepoint.
    """
    def __init__(self, device, nr_cpu):
        """Initializes the controller and its associated thread pool."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.nr_cpu = nr_cpu
        self.pool = ThreadPool(nr_cpu, device)

    def run(self):
        """The main simulation loop for the device controller."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # --- Termination logic ---
                # Send a 'None' task for each worker to signal it to exit.
                for _ in xrange(self.nr_cpu):
                    self.pool.add_task((True, None, None))
                # Wait for all worker threads to finish.
                self.pool.wait_workers()
                break

            # Wait for the supervisor to assign all scripts for this timepoint.
            self.device.time_point_done.wait()
            self.device.time_point_done.clear()

            # Add all assigned scripts as tasks to the thread pool's queue.
            for my_script in self.device.scripts:
                self.pool.add_task((False, my_script, neighbours))

            # Block until the workers have processed all tasks for this timepoint.
            self.pool.wait_completion()
            self.device.scripts = [] # Clear scripts for the next round.

            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()


class ThreadPool(object):
    """
    A simple thread pool manager that creates, manages, and provides tasks
    to a set of worker threads.
    """
    def __init__(self, num_threads, device):
        """Initializes the pool and starts the worker threads."""
        self.queue = Queue(num_threads)
        self.device = device
        self.workers = [AuxiliaryDeviceThread(self.device, self.queue) for _ in xrange(num_threads)]

    def add_task(self, info):
        """Adds a task to the shared queue for workers to process."""
        self.queue.put(info)

    def wait_completion(self):
        """Blocks until all items in the queue have been gotten and processed."""
        self.queue.join()

    def wait_workers(self):
        """Waits for all worker threads in the pool to terminate."""
        for adt in self.workers:
            adt.join()


class AuxiliaryDeviceThread(Thread):
    """A long-lived worker thread that consumes and executes tasks from a queue."""
    def __init__(self, device, queue):
        """Initializes the worker and sets it as a daemon."""
        Thread.__init__(self)
        self.queue = queue
        self.device = device
        self.daemon = True
        self.start()

    def run(self):
        """The main worker loop."""
        while True:
            # Block until a task is available in the queue.
            can_finish, got_script, neighbours = self.queue.get()

            # The 'can_finish' flag is the termination signal.
            if can_finish:
                break

            (script, location) = got_script

            # Acquire the specific lock for this location to ensure exclusive access.
            self.device.locations_lock_set[location].acquire()

            # --- Data Aggregation and Processing ---
            script_data = []
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                result = script.run(script_data)
                # Write results back; this is safe because the location lock is held.
                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)

            self.device.locations_lock_set[location].release()

            # Signal to the queue that this task is complete.
            self.queue.task_done()
