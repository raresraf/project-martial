
"""
This module provides a framework for simulating a distributed system of devices.

It includes classes for representing individual devices, managing their concurrent
execution, and a custom thread pool implementation for handling tasks. The simulation
appears to operate in discrete timepoints, with devices synchronizing at the end of
each timepoint using a barrier.
"""

from threading import Thread, Condition, Semaphore
from barrier import Barrier
from threadpool import ThreadPool

class Device(object):
    """Represents a single device in the distributed simulation.

    Each device runs in its own thread (`DeviceThread`), manages its own sensor data,
    and executes assigned scripts. It communicates and synchronizes with neighboring
    devices.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data.
        supervisor: An object responsible for providing network topology (neighbours).
        scripts (list): A list of scripts (tasks) to be executed.
        cond (Condition): A condition variable for synchronizing script assignment.
        barrier (Barrier): A barrier for synchronizing with other devices at the
                           end of a timepoint.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor: The supervisor object managing the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        # A semaphore per location to ensure thread-safe access to sensor data.
        self.data_semaphores = {loc : Semaphore(1) for loc in sensor_data}
        self.scripts = []

        # Flags for synchronization between the main thread and the device thread.
        self.new_script = False
        self.timepoint_end = False
        self.cond = Condition()

        self.barrier = None
        self.supervisor = supervisor
        self.thread = DeviceThread(self)

        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the synchronization barrier for all devices.

        This method should be called on one device (e.g., device_id 0) to act as
        the coordinator for setting up a shared barrier among all devices.
        
        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            # The device with ID 0 creates and distributes the barrier.
            self.barrier = Barrier(len(devices))
            for neigh in devices:
                if neigh.device_id != self.device_id:
                    neigh.set_barrier(self.barrier)

    def set_barrier(self, barrier):
        """Assigns a shared barrier to this device.

        Args:
            barrier (Barrier): The barrier object to be used for synchronization.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """Assigns a new script to be executed by the device.

        This method is called by an external entity (e.g., the supervisor) to
        give work to the device. It notifies the device's thread that new work
        is available.

        Args:
            script: The script object to be executed.
            location: The location context for the script execution.
        """
        with self.cond:
            if script is not None:
                self.scripts.append((script, location))
                self.new_script = True
            else:
                # A None script signals the end of a timepoint.
                self.timepoint_end = True
            self.cond.notifyAll()

    def timepoint_ended(self):
        """Blocks until the end of a timepoint is signaled.

        A timepoint ends when `assign_script` is called with a None script.
        This method is used by the `DeviceThread` to wait for all scripts for the
        current timepoint to be assigned.

        Returns:
            bool: True if the timepoint has ended, False if a new script arrived.
        """
        with self.cond:
            while not self.new_script and \
                  not self.timepoint_end:
                self.cond.wait()

            if self.new_script:
                self.new_script = False
                return False
            else:
                self.timepoint_end = False
                self.new_script = len(self.scripts) > 0
                return True

    def get_data(self, location):
        """Retrieves sensor data for a given location in a thread-safe manner.

        Args:
            location: The location for which to retrieve data.

        Returns:
            The sensor data at the given location, or None if the location
            is not found.
        """
        if location in self.sensor_data:
            self.data_semaphores[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data for a given location in a thread-safe manner.

        Args:
            location: The location to update.
            data: The new data to be stored.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_semaphores[location].release()

    def shutdown(self):
        """Shuts down the device thread."""
        self.thread.join()

class DeviceThread(Thread):
    """The main execution thread for a Device.
    
    This thread manages the device's lifecycle, including executing scripts
    within a timepoint and synchronizing with other devices at the end of it.
    """

    def __init__(self, device):
        """Initializes the device thread.
        
        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    @staticmethod
    def run_script(own_device, neighbours, script, location):
        """Executes a single script.

        This static method gathers data from the device itself and its neighbors,
        runs the script with the collected data, and then propagates the result
        back to all participating devices.
        
        Args:
            own_device (Device): The device executing the script.
            neighbours (list): A list of neighboring devices.
            script: The script to execute.
            location: The location context for data gathering.
        """
        script_data = []

        # Gather data from neighboring devices.
        for device in neighbours:
            if device is own_device:
                continue
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Gather data from the device itself.
        data = own_device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Execute the script with the aggregated data.
            result = script.run(script_data)

            # Distribute the result back to neighbors.
            for device in neighbours:
                if device is not own_device:
                    device.set_data(location, result)

            # Update the device's own data.
            own_device.set_data(location, result)

    def run(self):
        """The main loop of the device thread.

        It continuously processes scripts for each timepoint, using a thread pool
        for concurrent script execution, and synchronizes with a barrier at the
        end of each timepoint.
        """
        
        # Initialize a thread pool for executing scripts.
        pool_size = 8
        pool = ThreadPool(pool_size)

        while True:
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # If no neighbors, the simulation is likely ending.
                break

            # Process scripts for the current timepoint.
            offset = 0
            while not self.device.timepoint_ended():
                scripts = self.device.scripts[offset:]
                for (script, location) in scripts:
                    pool.add_task(DeviceThread.run_script, self.device,
                                  neighbours, script, location)
                
                offset = len(scripts)

            # Wait for all scripts in the current timepoint to complete.
            pool.wait()

            # Synchronize with all other devices before proceeding to the next timepoint.
            self.device.barrier.wait()

        # Cleanly shut down the thread pool.
        pool.terminate()


# The following classes seem to be a re-implementation of a thread pool,
# possibly because the code was intended for an older Python version or to
# avoid external dependencies.
from Queue import Queue
from threading import Thread

class Worker(Thread):
    """A worker thread that processes tasks from a queue.
    
    This is a component of the `ThreadPool` class.
    """

    def __init__(self, tasks):
        """Initializes the worker.
        
        Args:
            tasks (Queue): The queue from which to fetch tasks.
        """
        Thread.__init__(self)
        self.tasks = tasks

    def run(self):
        """The main loop for the worker thread.
        
        It continuously fetches tasks from the queue and executes them until a
        termination signal (ValueError) is received.
        """
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except ValueError:
                # A ValueError is used as a signal to terminate the worker.
                return
            finally:
                self.tasks.task_done()

class ThreadPool(object):
    """A simple thread pool implementation.

    It manages a queue of tasks and a set of worker threads to execute them.
    """

    def __init__(self, num_threads):
        """Initializes the thread pool.

        Args:
            num_threads (int): The number of worker threads to create.
        """
        self.tasks = Queue(num_threads)
        self.workers = [Worker(self.tasks) for _ in range(num_threads)]

        for worker in self.workers:
            worker.start()

    def add_task(self, func, *args, **kargs):
        """Adds a task to the queue to be executed.

        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            **kargs: Keyword arguments for the function.
        """
        self.tasks.put((func, args, kargs))

    def wait(self):
        """Blocks until all tasks in the queue have been processed."""
        self.tasks.join()

    def terminate(self):
        """Terminates all worker threads in the pool."""
        self.wait()

        def raising_dummy():
            """A dummy function that raises an exception to signal termination."""
            raise ValueError

        # Add a termination task for each worker.
        for _ in range(len(self.workers)):
            self.add_task(raising_dummy)
        for worker in self.workers:
            worker.join()
