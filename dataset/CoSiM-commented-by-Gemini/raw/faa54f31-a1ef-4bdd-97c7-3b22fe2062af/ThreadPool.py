"""
Implements a simulation framework for a network of distributed devices.

This module defines a custom `ThreadPool` for concurrent task execution and a `Device`
class that represents a node in the network. Each device runs its own thread and
communicates with its neighbors to execute scripts on distributed sensor data.
The simulation proceeds in synchronized timepoints, coordinated by a barrier.
"""

from Queue import Queue
from threading import Thread, Event, Lock

# Assuming 'barrier' module provides a Barrier class for synchronization.
from barrier import Barrier

class ThreadPool(object):
    """
    A thread pool to manage a number of worker threads for executing tasks.

    This pool is designed to work within the Device-centric simulation. Each worker
    thread pulls tasks from a shared queue, executes a script on data gathered
    from devices, and disseminates the results.

    Attributes:
        queue (Queue): The task queue for worker threads.
        threads (list): A list of the worker thread objects.
        device (Device): The parent device this thread pool is associated with.
    """

    def __init__(self, threads_count):
        """
        Initializes the ThreadPool.

        Args:
            threads_count (int): The number of worker threads to create.
        """
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None

        self.create_workers(threads_count)
        self.start_workers()

    def create_workers(self, threads_count):
        """Creates the worker threads."""
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

    def start_workers(self):
        """Starts the worker threads."""
        for thread in self.threads:
            thread.start()

    def set_device(self, device):
        """
        Associates the thread pool with a specific device.

        Args:
            device (Device): The parent device object.
        """
        self.device = device

    def execute(self):
        """
        The main execution loop for a worker thread.

        Continuously fetches tasks from the queue and executes them. A sentinel
        value of (None, None, None) in the queue signals the thread to terminate.
        """
        while True:
            neighbours, script, location = self.queue.get()

            # Sentinel check for thread termination
            if neighbours is None and script is None:
                self.queue.task_done()
                return

            self.run_script(neighbours, script, location)
            self.queue.task_done()

    def run_script(self, neighbours, script, location):
        """
        Executes a single script task.

        This function gathers data from the parent device and its neighbors,
        runs the script with the collected data, and then distributes the
        results back to all involved devices.

        Args:
            neighbours (list): A list of neighboring Device objects.
            script (Script): The script object to be executed.
            location (any): The data location identifier for this task.
        """
        script_data = []

        # Gather data from all neighbor devices.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Gather data from the local device itself.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Execute the script on the aggregated data.
            result = script.run(script_data)

            # Distribute the result back to the neighbors.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

            # Update the local device with the result.
            self.device.set_data(location, result)

    def submit(self, neighbours, script, location):
        """
        Submits a new task to the queue.

        Args:
            neighbours (list): A list of neighboring Device objects.
            script (Script): The script object to be executed.
            location (any): The data location identifier for this task.
        """
        self.queue.put((neighbours, script, location))

    def wait_threads(self):
        """Blocks until all items in the queue have been processed."""
        self.queue.join()

    def end_threads(self):
        """

        Shuts down all worker threads gracefully.

        First, it waits for the current queue to be empty. Then, it puts a
        sentinel value in the queue for each thread to signal termination.
        Finally, it joins all threads to wait for their completion.
        """
        self.wait_threads()

        # "Poison pill" for each thread to terminate its loop.
        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)

        for thread in self.threads:
            thread.join()

class Device(object):
    """
    Represents a single device in the distributed network simulation.

    Each device has a unique ID, local sensor data, and a control thread.
    It synchronizes with other devices using events and a shared barrier.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's local data.
        supervisor (Supervisor): An object responsible for providing neighbor info.
        barrier (Barrier): A synchronization primitive to sync all devices at the
                           end of a timepoint.
        location_locks (dict): A dictionary of locks to ensure thread-safe access
                               to sensor data.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device object.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for the device.
            supervisor (Supervisor): The supervisor object for the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.barrier = None
        self.location_locks = {location: Lock() for location in sensor_data}
        self.scripts_arrived = False

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the synchronization barrier for all devices.

        This method should be called on one device (conventionally device 0),
        which then creates and distributes the barrier to all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            self.send_barrier(devices, self.barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        """
        A static method to distribute the barrier to other devices.

        Args:
            devices (list): The list of all devices.
            barrier (Barrier): The shared barrier object.
        """
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """
        Sets the barrier for this device.

        Args:
            barrier (Barrier): The shared barrier object.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on this device for the current timepoint.

        Args:
            script (Script): The script to execute.
            location (any): The data location for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_arrived = True
        else:
            # A None script signals the end of script assignments for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safely retrieves data from a specific location.

        Args:
            location (any): The key for the data to retrieve.

        Returns:
            The data associated with the location, or None if the location
            is not found.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Thread-safely updates data at a specific location.

        Args:
            location (any): The key for the data to update.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()

    def shutdown(self):
        """Shuts down the device's control thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a Device.

    This thread manages the device's lifecycle within the simulation, including
    script processing, synchronization, and communication.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(8)

    def run(self):
        """The main simulation loop for the device."""
        self.thread_pool.set_device(self.device)

        # Main simulation loop continues as long as the supervisor provides neighbors.
        while True:
            # A "timepoint" starts here.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # End of simulation.
                break

            # Process all scripts for the current timepoint.
            while True:
                # Wait for scripts to be assigned or for a signal that assignment is done.
                if self.device.scripts_arrived or self.device.timepoint_done.wait():
                    if self.device.scripts_arrived:
                        self.device.scripts_arrived = False

                        # Submit all assigned scripts to the thread pool for execution.
                        for (script, location) in self.device.scripts:
                            self.thread_pool.submit(neighbours, script, location)
                    else:
                        # End of script assignments for this timepoint.
                        self.device.timepoint_done.clear()
                        self.device.scripts_arrived = True
                        break

            # Wait for all submitted scripts for this timepoint to complete.
            self.thread_pool.wait_threads()

            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()

        # Cleanly shut down the thread pool at the end of the simulation.
        self.thread_pool.end_threads()
