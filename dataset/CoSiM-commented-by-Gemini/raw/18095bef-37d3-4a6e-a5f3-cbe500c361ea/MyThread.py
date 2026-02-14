"""
This module defines a framework for a distributed device simulation.

It consists of three main classes:
- MyThread: A thread pool manager for executing tasks concurrently.
- Device: Represents a single device in the simulated network, with its own data and scripts.
- DeviceThread: The main control loop for a Device, managing its lifecycle and interaction
  with other devices and the simulation supervisor.

The architecture is designed for a time-stepped simulation where devices perform computations
based on their own data and the data of their neighbors, synchronized across each time step.
"""
from Queue import Queue
from threading import Thread, Event, Lock
from barrier import Barrier


class MyThread(object):
    """
    A thread pool manager that executes tasks from a queue.

    This class creates a fixed number of worker threads that continuously pull tasks
    from a shared queue and execute them. It is designed to process scripts
    on data gathered from a network of simulated devices.
    """

    def __init__(self, threads_count):
        """
        Initializes the thread pool.

        Args:
            threads_count (int): The number of worker threads to create.
        """
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None  # The parent device this thread pool belongs to.

        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

        for thread in self.threads:
            thread.start()

    def execute(self):
        """

        The main execution loop for a worker thread.

        A worker thread continuously waits for a task from the queue. A task is a tuple
        containing neighbors, a script, and a location. A sentinel value of
        (None, None, None) signals the thread to terminate.
        """
        while True:
            neighbours, script, location = self.queue.get()

            # Sentinel check for thread termination.
            if neighbours is None and script is None:
                self.queue.task_done()
                return

            self.run_script(neighbours, script, location)
            self.queue.task_done()

    def run_script(self, neighbours, script, location):
        """
        Executes a given script on data from the device and its neighbors.

        Args:
            neighbours (list): A list of neighboring Device objects.
            script (Script): The script object to be executed. It must have a `run` method.
            location (any): The data location identifier to be used for this script execution.
        """
        script_data = []

        # Gather data from all neighboring devices.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is None:
                    continue
                script_data.append(data)

        # Gather data from the current device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Execute the script on the collected data.
            result = script.run(script_data)

            # Propagate the result back to all neighbors.
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    continue
                device.set_data(location, result)

            # Set the result on the current device as well.
            self.device.set_data(location, result)

    def end_threads(self):
        """Gracefully shuts down all worker threads in the pool."""
        self.queue.join()  # Wait for all tasks to be completed.

        # Send a sentinel value for each thread to signal termination.
        for _ in xrange(len(self.threads)):
            self.queue.put((None, None, None))

        # Wait for all threads to finish.
        for thread in self.threads:
            thread.join()


class Device(object):
    """Represents a single device in the simulated network."""

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data.
            supervisor (Supervisor): The central supervisor object for the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        
        # Events for synchronization between the supervisor and device thread.
        self.script_received = Event()
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.new_adds()
        self.thread.start()

    def new_adds(self):
        """Initializes additional attributes for synchronization and data management."""
        self.barrier = None
        self.locations = {location: Lock() for location in self.sensor_data}
        self.script_arrived = False

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the synchronization barrier for all devices.
        This method should only be called on the root device (device_id == 0).
        """
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a new script to be executed by this device.

        Args:
            script (Script): The script to execute. If None, it signals the end of a timepoint.
            location (any): The data location for the script.
        """
        self.set_boolean(script)
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()  # Signal that a new script is ready.
        else:
            self.timepoint_done.set()  # Signal that the current timepoint is finished.

    def set_boolean(self, script):
        """Helper to set a flag indicating a script has arrived."""
        if script is not None:
            self.script_arrived = True

    def acquire_location(self, location):
        """Acquires the lock for a specific data location."""
        if location in self.sensor_data:
            self.locations[location].acquire()

    def get_data(self, location):
        """
        Thread-safely gets data from a specific location.

        Args:
            location (any): The location of the data to retrieve.

        Returns:
            The data at the given location, or None if the location doesn't exist.
        """
        self.acquire_location(location)
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Thread-safely sets data at a specific location.

        Args:
            location (any): The location to write to.
            data (any): The data to be written.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations[location].release()

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a Device."""

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = MyThread(8)  # Each device has its own pool of 8 worker threads.

    def run(self):
        """
        The main lifecycle of a device.

        This loop represents the device's operation over time. It synchronizes with
        other devices at each time step using a barrier.
        """
        self.thread_pool.device = self.device

        while True:
            # Get the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:  # A None value signals the end of the simulation.
                break

            # Inner loop for processing scripts within a single timepoint.
            while True:
                # Wait until a script arrives or the timepoint is signaled as done.
                if self.device.script_arrived or self.device.timepoint_done.wait():
                    if self.device.script_arrived:
                        self.device.script_arrived = False
                        # Put all received scripts into the thread pool's queue.
                        for (script, location) in self.device.scripts:
                            self.thread_pool.queue.put((neighbours, script, location))
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_arrived = True
                        break  # Exit inner loop to proceed to the barrier.

            # Wait for all scripts in the current timepoint to be processed.
            self.thread_pool.queue.join()

            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()

        # Clean up the thread pool at the end of the simulation.
        self.thread_pool.end_threads()
