"""
This module implements a simulation framework for a network of communicating devices.

It defines the `Device` class, which represents a node in the network, a `DeviceThread`
that runs the core logic for each device, and a `ThreadPool` for executing
computational scripts in parallel. The system appears designed for synchronous,
step-by-step simulations where devices exchange data with their neighbors,
process it, and then wait for all other devices to complete the step.
"""

from threading import Event, Thread, Lock

from barrier import ReusableBarrierCond
from threadpool import ThreadPool

class Device(object):
    """
    Represents a single device or node in a distributed simulation environment.

    Each device holds local sensor data, can be assigned computational scripts,
    and communicates with its neighbors to exchange data. Synchronization is
    managed through events and a reusable barrier to ensure that all devices
    operate in lock-step for each timepoint in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor values.
            supervisor (object): An external object that manages the network topology
                                 (e.g., providing lists of neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Event to signal that a new script has been assigned.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that a simulation timepoint is complete.
        self.timepoint_done = Event()

        # Create a dictionary of locks, one for each sensor data location,
        # to ensure thread-safe access to sensor data.
        self.locations_locks = []
        for location in sensor_data:
            self.locations_locks.append((location, Lock()))
        self.locations_locks = dict(self.locations_locks)

        # A reusable barrier for synchronizing with other devices at the end of a timepoint.
        self.barrier = None

        # The main thread that executes the device's logic loop.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the synchronization barrier for all devices in the simulation.

        This method should be called on one designated device (e.g., device_id 0)
        to create and distribute a shared barrier to all other devices.
        
        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Logic is centralized; only device 0 creates and distributes the barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a computational script to the device or signals the end of a timepoint.

        Args:
            script (object): The script to be executed. Should have a `run` method.
                             If None, it signals that the current timepoint is finished.
            location (str): The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script is the signal to end the current simulation step.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safely retrieves sensor data from a specific location.

        Args:
            location (str): The location from which to retrieve data.

        Returns:
            The sensor data at the given location, or None if the location
            is not monitored by this device.
        """
        if location in self.sensor_data:
            self.locations_locks[location].acquire()
            return self.sensor_data[location]

        return None


    def set_data(self, location, data):
        """

        Thread-safely updates sensor data at a specific location.
        Args:
            location (str): The location to update.
            data: The new data to be stored.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations_locks[location].release()


    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main worker thread for a Device.

    This thread manages the device's lifecycle, including fetching neighbors,
    processing scripts for each simulation timepoint, and synchronizing with
    other devices.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(8, device)

    def run(self):
        """The main execution loop for the device."""
        while True:
            # At the beginning of each major step, get the current set of neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbours signals the end of the simulation.
                break

            # Inner loop for processing one simulation timepoint.
            while True:

                # Wait until the timepoint is marked as done, but no new scripts have come in.
                # This is the condition to break and synchronize with other devices.
                if self.device.timepoint_done.wait() and not self.device.script_received.is_set():
                    self.device.timepoint_done.clear()
                    self.device.script_received.set() # Likely to prepare for the next round.
                    break

                # If new scripts have been received, process them.
                if self.device.script_received.is_set():
                    self.device.script_received.clear()

                    # Submit each assigned script to the thread pool for execution.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task(script, location, neighbours)


            # Wait until all tasks submitted to the thread pool are completed.
            self.thread_pool.tasks_queue.join()

            # Synchronize with all other devices. Waits here until all devices have
            # reached this barrier.
            self.device.barrier.wait()

        # Cleanly shut down the thread pool before exiting.
        self.thread_pool.join_threads()


from threading import Thread
from Queue import Queue

class ThreadPool(object):
    """
    A simple fixed-size thread pool for executing tasks.

    This pool is used by a Device to run computational scripts concurrently.
    """

    def __init__(self, number_threads, device):
        """
        Initializes the ThreadPool.

        Args:
            number_threads (int): The number of worker threads to create.
            device (Device): The parent device, used for context.
        """
        self.number_threads = number_threads
        self.device_threads = []
        self.device = device
        self.tasks_queue = Queue(number_threads)


        # Create and start the worker threads.
        for _ in xrange(0, number_threads):
            thread = Thread(target=self.apply_scripts)
            self.device_threads.append(thread)

        for thread in self.device_threads:
            thread.start()

    def apply_scripts(self):
        """The target function for worker threads."""
        while True:
            # Wait for a task from the queue.
            script, location, neighbours = self.tasks_queue.get()

            # A task of (None, None, None) is a sentinel value to signal thread termination.
            if neighbours is None and script is None:
                self.tasks_queue.task_done()
                return

            script_data = []
            # Gather data from all neighboring devices for the specified location.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Also gather data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Execute the script with the collected data.
                result = script.run(script_data)

                # Distribute the result back to all neighbors.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                # Also update the local device's data.
                self.device.set_data(location, result)

            # Signal that the current task is complete.
            self.tasks_queue.task_done()


    def submit_task(self, script, location, neighbours):
        """
        Adds a new task to the queue for the worker threads to execute.

        Args:
            script (object): The script to run.
            location (str): The data location to work on.
            neighbours (list): A list of neighboring devices.
        """
        self.tasks_queue.put((script, location, neighbours))


    def join_threads(self):
        """Shuts down all worker threads in the pool."""
        # Wait for all submitted tasks to be completed.
        self.tasks_queue.join()

        # Send a sentinel value (None, None, None) for each thread to signal it to exit.
        for _ in xrange(0, len(self.device_threads)):
            self.submit_task(None, None, None)

        # Wait for all worker threads to terminate.
        for thread in self.device_threads:
            thread.join()