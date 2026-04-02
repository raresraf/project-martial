
"""
This module simulates a distributed system of devices that concurrently process
sensor data in a synchronized, time-stepped manner. It models a Single Program,
Multiple Data (SPMD) architecture where devices execute scripts on local and
neighboring data.

The core components are:
- Device: Represents a node in the system, managing its own data and a thread pool.
- DeviceThread: The main control loop for a device, synchronizing with other
  devices using a barrier.
- ThreadPool and Worker: Manage and execute computational tasks in parallel,
  ensuring data consistency through location-based locks.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from thread_pool import ThreadPool

class Device(object):
    """
    Represents a single device in the distributed system.

    Each device has its own sensor data and executes scripts in coordination
    with other devices. It uses a thread pool to perform computations in parallel.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor data,
                                keyed by location.
            supervisor (Supervisor): An external entity that coordinates the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Event to signal that a new timepoint has been reached and scripts can be run.
        self.timepoint_done = Event()

        # A reusable barrier for synchronizing all devices at the end of a timepoint.
        self.barrier = None

        # A dictionary of locks, one for each data location, shared across all devices.
        self.location_locks = {}

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the synchronization mechanisms for a group of devices.

        This method sets up a shared barrier and a set of shared locks for all
        devices in the simulation, ensuring they can coordinate access to data
        at shared locations.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.barrier is None:
            # Create a barrier for all devices to synchronize on.
            self.barrier = ReusableBarrierCond(len(devices))

            # Share the barrier and location locks with all other devices.
            for device in devices:
                device.barrier = self.barrier

                # Create a lock for each unique sensor data location across all devices.
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
                
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a specific location.

        Args:
            script (object): The script object with a `run` method to be executed.
            location (any): The location key for the data the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is a signal that the timepoint processing is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.

        Args:
            location (any): The location of the data to retrieve.

        Returns:
            The data at the given location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """

        Updates the sensor data at a specific location.

        Args:
            location (any): The location of the data to update.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.
    
    This thread orchestrates the device's participation in the time-stepped
    simulation, including task submission and synchronization.
    """
    
    # Defines the number of worker threads in the device's thread pool.
    NO_CORES = 8

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.thread_pool = ThreadPool(self.device, DeviceThread.NO_CORES)

    def run(self):
        """The main loop of the device thread, executed for the duration of the simulation."""
        while True:
            
            # Get the set of neighboring devices for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbours signals the end of the simulation.
                # Submit sentinel values to terminate the worker threads.
                for _ in xrange(DeviceThread.NO_CORES):
                    self.thread_pool.submit_task(None, None, None)
                
                self.thread_pool.end_workers()
                break

            # Wait until the supervisor signals that a new timepoint has begun.
            self.device.timepoint_done.wait()

            # Submit all assigned scripts as tasks to the thread pool for execution.
            for (script, location) in self.device.scripts:
                self.thread_pool.submit_task(script, location, neighbours)

            # Clear the event to prepare for the next timepoint.
            self.device.timepoint_done.clear()

            # Wait at the barrier for all other devices to finish their timepoint processing.
            # This ensures all devices move to the next timepoint in lock-step.
            self.device.barrier.wait()




from threading import Thread
from Queue import Queue

class Worker(Thread):
    """
    A worker thread that executes computational tasks for a device.

    Workers pull tasks from a shared queue and execute them, handling data
    gathering, script execution, and result dissemination.
    """

    def __init__(self, device, task_queue):
        """
        Initializes a Worker thread.

        Args:
            device (Device): The parent device.
            task_queue (Queue): The queue from which to pull tasks.
        """
        Thread.__init__(self)
        self.device = device
        self.task_queue = task_queue

    def run(self):
        """Continuously processes tasks from the queue until a sentinel is received."""
        while True:
            # Dequeue a task. This call blocks until a task is available.
            script, location, neighbours = self.task_queue.get()

            # Sentinel check: A (None, None, None) tuple signals thread termination.
            if (script is None and location is None and neighbours is None):
                self.task_queue.task_done()
                break

            # Acquire the lock for the specific data location to ensure exclusive access.
            # This prevents race conditions when multiple devices/workers access the
            # same data location concurrently.
            with self.device.location_locks[location]:
                
                self.run_task(script, location, neighbours)

            # Signal that the task is complete.
            self.task_queue.task_done()

    def run_task(self, script, location, neighbours):
        """
        Executes a single computational task.

        This involves gathering data from the parent device and its neighbors,
        running the script, and updating the data on all involved devices.
        """
        script_data = []
        
        # Gather data from neighboring devices at the specified location.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)


        # Only run the script if there is data to process.
        if script_data != []:
            
            result = script.run(script_data)

            # Disseminate the result to all neighboring devices.
            # This updates the data at the location for the entire neighborhood.
            for device in neighbours:
                device.set_data(location, result)

            self.device.set_data(location, result)


class ThreadPool(object):
    """
    Manages a pool of worker threads for a device.
    """

    def __init__(self, device, no_workers):
        """
        Initializes the ThreadPool.

        Args:
            device (Device): The parent device.
            no_workers (int): The number of worker threads to create.
        """
        self.device = device
        self.no_workers = no_workers
        
        self.task_queue = Queue(no_workers)
        self.workers = []
        self.initialize_workers()

    def initialize_workers(self):
        """Creates and starts the worker threads."""
        for _ in xrange(self.no_workers):
            self.workers.append(Worker(self.device, self.task_queue))

        for worker in self.workers:
            worker.start()

    def end_workers(self):
        """Waits for all tasks to be completed and joins all worker threads."""
        # Wait for the queue to be empty.
        self.task_queue.join()

        # Join each worker thread to ensure a clean shutdown.
        for worker in self.workers:
            worker.join()

    def submit_task(self, script, location, neighbours):
        """
        Adds a new task to the task queue for the workers to process.

        Args:
            script: The script to execute.
            location: The data location for the script.
            neighbours: The neighboring devices involved in the task.
        """
        self.task_queue.put((script, location, neighbours))
