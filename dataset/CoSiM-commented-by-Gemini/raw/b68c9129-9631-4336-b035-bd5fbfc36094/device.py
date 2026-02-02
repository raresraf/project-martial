



"""
This module implements a simulation of a network of interconnected devices.

The simulation is built around three core classes:
- Device: Represents a node in the network, managing its sensor data and state.
- DeviceThread: The main control thread for a device, handling the simulation
  loop and interaction with a thread pool.
- ThreadPool: Manages a pool of worker threads to execute tasks concurrently.

The system uses a combination of barriers and events for synchronization across
devices and threads, allowing for discrete time-step simulation.
"""

from threading import Event, Thread, Lock
from barrier import Barrier
from thread_pool import ThreadPool


class Device(object):
    """
    Represents a single device in the simulation environment.

    Each device has a unique ID, local sensor data, and communicates with a
    central supervisor. It processes tasks (scripts) using a dedicated thread
    that manages a thread pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local
                                sensor readings, keyed by location.
            supervisor (Supervisor): The central supervisor object that manages
                                     the network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data


        self.supervisor = supervisor

        self.barrier = None
        self.scripts = []
        self.timepoint_done = Event()

        # Creates a lock for each sensor data location to ensure thread-safe access.
        self.locks = {}

        for location in sensor_data:
            self.locks[location] = Lock()

        # Flag to indicate if new scripts have been assigned in a time step.
        self.scripts_available = False

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes the synchronization barrier to all devices.

        This method is intended to be called on a single master device (ID 0)
        to create a barrier for all devices in the simulation.
        """
        if self.device_id == 0:
            barrier = Barrier(len(devices))
            self.barrier = barrier
            self.send_barrier(devices, barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        """
        A static method to distribute the barrier to all other devices.

        Args:
            devices (list): The list of all devices in the simulation.
            barrier (Barrier): The shared barrier instance.
        """
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """
        Sets the synchronization barrier for this device.

        Args:
            barrier (Barrier): The shared barrier instance.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed in the next time step.

        A script value of None is a signal that the current time point is complete.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location in a thread-safe manner.

        Acquires a lock for the location before reading.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]

        return None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location and releases the lock.

        Assumes that the lock for the location has been previously acquired.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread of execution for a device.

    This thread manages the device's lifecycle in the simulation, including
    handling time steps, interacting with the thread pool, and synchronizing
    with other devices.
    """
    NR_THREADS = 8

    def __init__(self, device):
        """
        Initializes the device thread and its associated thread pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        self.thread_pool = ThreadPool(self.NR_THREADS)

    def run(self):
        """The main simulation loop for the device."""
        self.thread_pool.set_device(self.device)

        while True:
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbours signals the end of the simulation.
                break

            while True:
                # Wait for the signal that the current time point is defined.
                self.device.timepoint_done.wait()
                if self.device.scripts_available:
                    self.device.scripts_available = False

                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task((neighbours, location, script))
                else:
                    # If no scripts are available, the time step work is done.
                    self.device.timepoint_done.clear()
                    self.device.scripts_available = True
                    break

            # Wait for all tasks in the thread pool to complete for this step.
            self.thread_pool.wait()

            # Wait at the barrier for all other devices to complete the step.
            self.device.barrier.wait()

        # Cleanly shut down the thread pool.
        self.thread_pool.finish()


from threading import Thread
from Queue import Queue


class ThreadPool(object):
    """
    A simple thread pool implementation for executing tasks.
    """
    def __init__(self, nr_threads):
        """
        Initializes the thread pool with a fixed number of worker threads.
        """


        self.device = None

        self.queue = Queue(nr_threads)
        self.thread_list = []

        self.create_threads(nr_threads)
        self.start_threads()

    def create_threads(self, nr_threads):
        """Creates the worker threads."""
        for _ in xrange(nr_threads):
            thread = Thread(target=self.execute_task)
            self.thread_list.append(thread)

    def start_threads(self):
        """Starts the worker threads."""
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].start()

    def set_device(self, device):
        """Associates the thread pool with a device."""
        self.device = device

    def submit_task(self, task):
        """Adds a task to the execution queue."""
        self.queue.put(task)

    def execute_task(self):
        """The target function for worker threads; continuously pulls and runs tasks."""

        while True:
            task = self.queue.get()

            neighbours = task[0]
            script = task[2]

            # A None script is a signal for the thread to terminate.
            if script is None and neighbours is None:
                self.queue.task_done()
                break

            self.run_script(task)
            self.queue.task_done()

    def run_script(self, task):
        """
        Executes a single script task.

        This involves gathering data from neighbor devices, running the script,
        and then propagating the results back.
        """

        neighbours, location, script = task
        script_data = []

        # Gather data from neighboring devices.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Gather data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Execute the script with the collected data.
            result = script.run(script_data)

            # Distribute the result to all neighboring devices.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

            # Update the local device with the result.
            self.device.set_data(location, result)

    def wait(self):
        """Blocks until all tasks in the queue are processed."""
        self.queue.join()

    def finish(self):
        """Shuts down the thread pool, waiting for all tasks to complete."""
        self.wait()

        # Send a termination signal (None task) to each worker thread.
        for _ in xrange(len(self.thread_list)):
            self.submit_task((None, None, None))

        # Wait for all worker threads to join.
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].join()
