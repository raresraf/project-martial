"""
This module implements a distributed device simulation framework using Python's
threading capabilities. It models a network of devices that operate on shared
sensor data, executing custom scripts in a synchronized, time-step-based manner.
The architecture uses a main thread per device, a worker pool for parallel
script execution, and synchronization primitives like barriers and locks to
ensure data consistency and coordinated progression through simulation timepoints.
"""

from threading import Lock, Event, Thread
from barrier import ReusableBarrierCond
from workerpool import WorkerPool


class Device(object):
    """
    Represents a single device in the distributed simulation.

    Each device maintains its own sensor data and executes assigned scripts.
    It coordinates with other devices through a shared barrier and a set of
    locks for accessing data at different locations, ensuring that operations
    are synchronized across the entire system.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor readings,
                            keyed by location.
        supervisor (object): A central supervisor object that manages the
                             overall simulation and device interactions.
        scripts (list): A list of (script, location) tuples to be executed
                        in the current time step.
        timepoint_done (Event): An event that signals the device has received
                                its scripts for the current time step.
        barrier (ReusableBarrierCond): A barrier for synchronizing all devices
                                       at the end of a time step.
        locks (dict): A dictionary of locks, keyed by location, to protect
                      access to sensor_data.
        thread (DeviceThread): The main execution thread for this device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor (object): The simulation supervisor.
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
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes synchronization primitives for all devices.

        This method should be called on one device (e.g., device_id 0) to create
        and share the barrier and location-based locks among all devices in the
        simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # This setup is centralized and performed by a single device (ID 0)
        # to ensure all devices share the same synchronization objects.
        if self.device_id == 0:
            num_threads = len(devices)
            barrier = ReusableBarrierCond(num_threads)
            location_locks = {}

            # Create a lock for each unique data location across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if location not in location_locks:
                        location_locks[location] = Lock()

            # Assign the shared barrier and locks to this device.
            if self.barrier is None:
                self.barrier = barrier
                self.locks = location_locks

            # Distribute the shared barrier and locks to all other devices.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
                if device.locks is None:
                    device.locks = location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device in the current time step.

        If the script is None, it signals that the simulation for this time
        step is complete for this device.

        Args:
            script (object): The script to be executed. Should have a `run` method.
            location (str): The location associated with the script's execution.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is a signal from the supervisor to proceed.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location from which to retrieve data.

        Returns:
            The data at the given location, or None if the location is not found.
        """
        data = None

        if location in self.sensor_data:
            data = self.sensor_data[location]
        return data

    def set_data(self, location, data):
        """
        Updates sensor data for a specific location.

        Args:
            location (str): The location to update.
            data: The new data to be stored.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's execution thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device.

    This thread orchestrates the device's participation in the simulation's
    time steps. It waits for scripts, submits them to a worker pool for
    execution, and synchronizes with other devices using a barrier.

    Attributes:
        device (Device): The device this thread belongs to.
        pool (WorkerPool): The worker pool used for executing scripts.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.pool = WorkerPool(8, self.device) # Creates a pool of 8 worker threads.

    def run(self):
        """The main loop of the device thread."""
        while True:
            # Get the list of neighbors from the supervisor for the current time step.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbours is the signal to shut down.
                # Submit sentinel values (None) to the worker pool to terminate workers.
                for _ in xrange(self.pool.max_threads):
                    self.pool.submit_work(None, None, None)
                self.pool.wait_completion()
                break

            # Wait for the supervisor to assign all scripts for this time point.
            self.device.timepoint_done.wait()

            # Submit all assigned scripts to the worker pool for execution.
            for (script, location) in self.device.scripts:
                self.pool.submit_work(neighbours, script, location)

            # Wait for all submitted scripts to complete execution.
            self.pool.wait_completion()

            # Synchronize with all other devices at the barrier before proceeding
            # to the next time step.
            self.device.barrier.wait()

            # Reset the event for the next time step.
            self.device.timepoint_done.clear()

        # Clean up the worker pool threads upon shutdown.
        self.pool.end_threads()


from threading import Thread
from Queue import Queue


class WorkerPool(object):
    """
    A pool of worker threads for executing tasks in parallel.

    This class manages a fixed-size pool of WorkerThread instances and uses a
    queue to distribute tasks (script executions) among them.

    Attributes:
        max_threads (int): The number of worker threads in the pool.
        queue (Queue): The queue used to hold tasks for the workers.
        device (Device): The parent device.
        thread_list (list): A list of the WorkerThread instances.
    """

    def __init__(self, no_workers, device):
        """
        Initializes the WorkerPool.

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
        """
        Submits a new task to the worker pool's queue.

        Args:
            neighbours (list): A list of neighboring devices.
            script (object): The script to be executed.
            location (str): The location for the script execution.
        """
        self.queue.put((neighbours, script, location))

    def wait_completion(self):
        """Blocks until all tasks in the queue have been processed."""
        self.queue.join()

    def end_threads(self):
        """Waits for all worker threads to terminate."""
        for thread in self.thread_list:
            thread.join()
        self.thread_list = []


class WorkerThread(Thread):
    """
    A worker thread that executes scripts.

    It continuously fetches tasks from a shared queue. For each task, it
    acquires a lock for the specified location, gathers data from the parent
    device and its neighbors, runs the script, and updates the data.

    Attributes:
        device (Device): The parent device.
        tasks (Queue): The queue from which to fetch tasks.
    """

    def __init__(self, device, tasks):
        """
        Initializes a WorkerThread.

        Args:
            device (Device): The parent device.
            tasks (Queue): The shared task queue.
        """
        Thread.__init__(self)
        self.device = device
        self.tasks = tasks

    def run(self):
        """The main loop of the worker thread."""
        while True:
            # Get a task from the queue. This is a blocking call.
            neighbours, script, location = self.tasks.get()

            # Check for the sentinel value (None, None, None) which signals termination.
            if neighbours is None and script is None and location is None:
                self.tasks.task_done()
                return

            # Acquire the lock for the given location to ensure exclusive access to data.
            with self.device.locks[location]:
                script_data = []
                
                # Gather data from all neighboring devices at the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the parent device itself.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Update the data on all neighboring devices with the new result.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Update the data on the parent device.
                    self.device.set_data(location, result)

            # Signal that the task is complete.
            self.tasks.task_done()
