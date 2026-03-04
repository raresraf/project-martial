"""
Implements a thread pool-based simulation of a distributed device network.

This script provides a more advanced simulation framework compared to a simple
thread-per-script model. It defines a `ThreadPool` for efficient management of
concurrent script executions and a `Device` class that utilizes this pool. This
design is suitable for simulating IoT or sensor networks where devices need to
process multiple tasks concurrently.
"""

from Queue import Queue
from threading import Thread

class ThreadPool(object):
    """
    A generic thread pool for executing tasks concurrently.

    This thread pool uses a queue to manage tasks and a fixed number of worker
    threads to execute them. It is designed to be used in the device simulation
    to handle the execution of multiple scripts in parallel.
    """

    def __init__(self, threads_count):
        """
        Initializes the thread pool with a fixed number of worker threads.

        Args:
            threads_count: The number of worker threads to create.
        """

        self.queue = Queue(threads_count)

        self.threads = []
        self.device = None

        self.create_workers(threads_count)
        self.start_workers()

    def create_workers(self, threads_count):
        """
        Creates the worker threads for the pool.

        Args:
            threads_count: The number of worker threads to create.
        """

        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

    def start_workers(self):
        """Starts all the worker threads in the pool."""

        for thread in self.threads:


            thread.start()

    def set_device(self, device):
        """
        Sets the parent device for the thread pool.

        This is used to give context to the worker threads when they are
        executing scripts.

        Args:
            device: The parent Device object.
        """
        self.device = device

    def execute(self):
        """
        The main execution loop for a worker thread.

        A worker thread continuously fetches tasks from the queue and executes
        them. A task is a tuple of (neighbours, script, location). A special
        task (None, None, None) is used to signal the thread to terminate.
        """

        while True:

            neighbours, script, location = self.queue.get()

            # Shutdown signal
            if neighbours is None and script is None:
                self.queue.task_done()
                return

            self.run_script(neighbours, script, location)
            self.queue.task_done()

    def run_script(self, neighbours, script, location):
        """
        Executes a single script.

        This method gathers data from neighboring devices and the parent device,
        runs the script with the collected data, and then disseminates the
        result back to the devices.

        Args:
            neighbours: A list of neighboring Device objects.
            script: The script to be executed.
            location: The location at which the script is to be executed.
        """

        script_data = []

        # Gather data from neighbors.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Gather data from the current device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Run the script with the collected data.
            result = script.run(script_data)

            # Distribute the result to neighbors.
            for device in neighbours:
                if device.device_id != self.device.device_id:


                    device.set_data(location, result)

            # Set the result for the current device.
            self.device.set_data(location, result)

    def submit(self, neighbours, script, location):
        """
        Submits a new task to the thread pool's queue.

        Args:
            neighbours: A list of neighboring Device objects.
            script: The script to be executed.
            location: The location for the script's execution.
        """

        self.queue.put((neighbours, script, location))

    def wait_threads(self):
        """Blocks until all tasks in the queue have been processed."""

        self.queue.join()

    def end_threads(self):
        """Shuts down the thread pool gracefully."""

        self.wait_threads()

        # Send a shutdown signal to each worker thread.
        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)

        for thread in self.threads:
            thread.join()




from threading import Event, Thread, Lock

from barrier import Barrier
from ThreadPool import ThreadPool

class Device(object):
    """
    Represents a device in the simulation that uses a ThreadPool for script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id: Unique ID for the device.
            sensor_data: Local sensor data for the device.
            supervisor: A reference to the simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()


        self.scripts = []
        self.timepoint_done = Event()

        self.barrier = None
        self.location_locks = {location : Lock() for location in sensor_data}
        self.scripts_arrived = False

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the barrier for synchronization, performed by device 0.

        Args:
            devices: A list of all devices in the simulation.
        """
        if self.device_id == 0:


            self.barrier = Barrier(len(devices))
            self.send_barrier(devices, self.barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        """
        Distributes the shared barrier to all other devices.

        Args:
            devices: The list of all devices.
            barrier: The shared Barrier object.
        """

        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """
        Sets the shared barrier for this device.

        Args:
            barrier: The shared Barrier object.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        Args:
            script: The script to be executed.
            location: The location of execution.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_arrived = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safely retrieves data from a specific location.

        Args:
            location: The location to retrieve data from.

        Returns:
            The data at the location, or None if not found.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Thread-safely sets data at a specific location.

        Args:
            location: The location to set data at.
            data: The data to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a Device, managing a ThreadPool for tasks.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The parent Device object.
        """


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        self.thread_pool = ThreadPool(8)

    def run(self):
        """
        The main execution loop for the device.

        This loop continuously gets neighbors, processes scripts using the
        thread pool, and synchronizes with other devices at a barrier.
        """
        self.thread_pool.set_device(self.device)

        while True:

            # Get neighbors for the current simulation step.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for and process scripts for the current timepoint.
            while True:

                # The device processes scripts either when they have arrived or
                # when the timepoint_done event is set.
                if self.device.scripts_arrived or self.device.timepoint_done.wait():
                    if self.device.scripts_arrived:
                        self.device.scripts_arrived = False

                        # Submit all scripts to the thread pool.
                        for (script, location) in self.device.scripts:
                            self.thread_pool.submit(neighbours, script, location)
                    else:
                        self.device.timepoint_done.clear()
                        self.device.scripts_arrived = True
                        break

            # Wait for all scripts in the current batch to complete.
            self.thread_pool.wait_threads()

            # Synchronize with all other devices.
            self.device.barrier.wait()

        # Cleanly shut down the thread pool.
        self.thread_pool.end_threads()