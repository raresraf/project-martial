"""
This module implements a distributed device simulation using a clean, reusable
worker pool architecture to process computational scripts in synchronized time
steps. It demonstrates a clear separation of concerns between device state,
simulation flow, and task execution.
"""

from threading import Lock, Event, Thread, Queue
from barrier import ReusableBarrierCond
from workerpool import WorkerPool


class Device(object):
    """
    Represents a single device in the simulation network.

    This class holds the device's state, including its sensor data and assigned
    scripts, and owns the main control thread (`DeviceThread`).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary of initial sensor data, keyed
                                by location.
            supervisor: The external entity managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        self.barrier = None
        self.locks = None

        self.thread = DeviceThread(self)
        self.thread.start()


    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.

        Device 0 acts as a coordinator, creating a single shared barrier and a
        shared dictionary of per-location locks. These objects are then
        distributed to all other devices in the simulation, ensuring they all
        synchronize on the same primitives.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            num_threads = len(devices)
            barrier = ReusableBarrierCond(num_threads)
            location_locks = {}

            # Create a unique lock for each location found across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if location not in location_locks:
                        location_locks[location] = Lock()

            # Distribute the shared objects to all devices.
            if self.barrier is None:
                self.barrier = barrier
                self.locks = location_locks
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
                if device.locks is None:
                    device.locks = location_locks

    def assign_script(self, script, location):
        """
        Assigns a script from the supervisor for the current time step.

        A `None` script signals that all assignments for the time step are complete.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Gets sensor data for a location.

        Note: This method is not thread-safe. Locking is handled by the
        `WorkerThread` that calls this method.
        """
        data = None
        if location in self.sensor_data:
            data = self.sensor_data[location]
        return data

    def set_data(self, location, data):
        """
        Sets sensor data for a location.

        Note: This method is not thread-safe. Locking is handled by the
        `WorkerThread` that calls this method.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    It orchestrates the simulation's time steps by managing a `WorkerPool` and
    handling high-level synchronization.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Creates a reusable pool of 8 worker threads.
        self.pool = WorkerPool(8, self.device)

    def run(self):
        """
        The main simulation loop.
        
        In each time step, it waits for scripts, submits them to the worker
        pool, waits for their completion, and synchronizes with other devices.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Shutdown signal from supervisor.
                # Send poison pill to all workers.
                for _ in xrange(self.pool.max_threads):
                    self.pool.submit_work(None, None, None)
                self.pool.wait_completion()
                break

            # Wait for the supervisor to signal all scripts have been assigned.
            self.device.timepoint_done.wait()

            # Submit all assigned scripts to the worker pool.
            for (script, location) in self.device.scripts:
                self.pool.submit_work(neighbours, script, location)

            # Wait for all workers to finish their tasks for this time step.
            self.pool.wait_completion()

            # Synchronize with all other devices before starting the next time step.
            self.device.barrier.wait()

            # Reset the event for the next time step.
            self.device.timepoint_done.clear()

        # Cleanly join all worker threads on shutdown.
        self.pool.end_threads()


class WorkerPool(object):
    """
    A classic worker pool that manages a task queue and a set of workers.
    """

    def __init__(self, no_workers, device):
        """
        Initializes the pool and starts the worker threads.

        Args:
            no_workers (int): The number of worker threads to create.
            device (Device): The parent device.
        """
        self.max_threads = no_workers
        self.queue = Queue(no_workers)
        self.device = device
        self.thread_list = []
        for _ in range(no_workers):
            thread = WorkerThread(self.device, self.queue)
            self.thread_list.append(thread)
            thread.start()

    def submit_work(self, neighbours, script, location):
        """Adds a new task to the work queue."""
        self.queue.put((neighbours, script, location))

    def wait_completion(self):
        """Blocks until all items in the queue have been processed."""
        self.queue.join()

    def end_threads(self):
        """Waits for all worker threads to terminate."""
        for thread in self.thread_list:
            thread.join()
        self.thread_list = []


class WorkerThread(Thread):
    """
    A persistent worker thread that executes tasks from a shared queue.
    """

    def __init__(self, device, tasks):
        """
        Initializes the worker.

        Args:
            device (Device): The parent device.
            tasks (Queue): The shared work queue.
        """
        Thread.__init__(self)
        self.device = device
        self.tasks = tasks

    def run(self):
        """
        The main loop for the worker.
        
        It continuously fetches tasks and executes them. Synchronization for
        data access is handled here using a per-location lock.
        """
        while True:
            # Blocks until a task is available.
            neighbours, script, location = self.tasks.get()

            # A "poison pill" of (None, None, None) signals termination.
            if neighbours is None and script is None and location is None:
                self.tasks.task_done()
                return

            # Acquire the specific lock for the location being worked on.
            with self.device.locks[location]:
                # Read phase.
                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Compute and Write phase.
                if script_data != []:
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

            # Signal that the task is complete.
            self.tasks.task_done()
