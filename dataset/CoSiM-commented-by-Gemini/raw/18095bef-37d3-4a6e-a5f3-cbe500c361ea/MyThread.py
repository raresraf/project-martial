"""
This module defines a multi-threaded framework for a distributed device simulation.
It includes a thread pool manager, a Device class representing a node in the
simulation, and a DeviceThread class that controls the execution cycle of a device.
"""
from Queue import Queue
from threading import Thread

class MyThread(object):
    """
    A thread pool manager that executes tasks from a shared queue.
    
    This class creates a fixed number of worker threads that continuously pull
    tasks from a queue and execute them. It is designed to process scripts
    on data from a network of devices.
    """

    def __init__(self, threads_count):
        """
        Initializes the thread pool.
        
        Args:
            threads_count (int): The number of worker threads to create.
        """
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None  # The device this thread pool belongs to.

        # Create and start the worker threads.
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

        for thread in self.threads:
            thread.start()

    def execute(self):
        """
        The target function for worker threads.
        
        Continuously fetches tasks from the queue and executes them until a
        sentinel value (None, None, None) is received.
        """
        while True:
            # A task is a tuple containing neighbours, a script, and a location.
            neighbours, script, location = self.queue.get()

            # Sentinel check: A None value for neighbours indicates a shutdown signal.
            if neighbours is None:
                if script is None:
                    self.queue.task_done()
                    return # Exit the thread loop.

            # Execute the script with the provided data.
            self.run_script(neighbours, script, location)
            self.queue.task_done()

    def run_script(self, neighbours, script, location):
        """
        Gathers data, runs a script, and disseminates the results.
        
        Args:
            neighbours (list): A list of neighboring Device objects.
            script (Script): The script object to be executed.
            location (any): The data location identifier to work on.
        """
        script_data = []

        # Aggregate data from all neighboring devices.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is None:
                    continue
                script_data.append(data)

        # Aggregate data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Run the script on the aggregated data.
            result = script.run(script_data)

            # Distribute the result back to all neighbors.
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    continue
                device.set_data(location, result)

            # Set the result on the local device as well.
            self.device.set_data(location, result)

    def end_threads(self):
        """
        Gracefully shuts down all worker threads in the pool.
        """
        # Wait for all tasks in the queue to be completed.
        self.queue.join()

        # Send a sentinel value to each thread to signal termination.
        for _ in xrange(len(self.threads)):
            self.queue.put((None, None, None))

        # Wait for all threads to finish execution.
        for thread in self.threads:
            thread.join()

from threading import Event, Thread, Lock
from barrier import Barrier
# MyThread is imported locally, assuming it's in the same directory.
# from MyThread import MyThread


class Device(object):
    """
    Represents a single device (node) in the distributed simulation.
    
    Each device has its own sensor data, a unique ID, and a thread to manage
    its lifecycle. It synchronizes with other devices using barriers and events.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device object.
        
        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data.
            supervisor (Supervisor): The central supervisor object managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal script arrival.
        self.scripts = []
        self.timepoint_done = Event() # Event to signal completion of a timepoint.
        self.thread = DeviceThread(self)

        self.new_adds()

        self.thread.start()

    def new_adds(self):
        """Initializes additional attributes for synchronization and data management."""
        self.barrier = None # Barrier for cross-device synchronization.
        self.locations = {}
        # Create a lock for each data location to ensure thread-safe access.
        for location in self.sensor_data:
            self.locations[location] = Lock()
        self.script_arrived = False

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the synchronization barrier for all devices.
        
        This method is intended to be called by a master device (device_id 0).
        """
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            for dev in devices:
                if dev.device_id == 0:
                    continue
                dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device.
        
        Args:
            script (Script): The script to execute. If None, it signals the
                             end of a timepoint.
            location (any): The location associated with the script.
        """
        self.set_boolean(script)
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def set_boolean(self, script):
        """Sets a flag indicating that a script has arrived."""
        if script is not None:
            self.script_arrived = True

    def acquire_location(self, location):
        """Acquires the lock for a specific data location."""
        if location in self.sensor_data:
            self.locations[location].acquire()

    def get_data(self, location):
        """
        Thread-safely gets data from a specific location.
        
        Returns:
            The data at the given location, or None if the location doesn't exist.
        """
        self.acquire_location(location)
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Thread-safely sets data at a specific location and releases the lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations[location].release()

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.
    
    This thread manages the device's lifecycle, including fetching neighbors,
    processing scripts for each timepoint, and synchronizing with other devices.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.
        
        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Each device has its own thread pool for executing scripts.
        self.thread_pool = MyThread(8)

    def run(self):
        """The main simulation loop for the device."""
        self.thread_pool.device = self.device

        while True:
            # Get the list of neighbors for the current simulation phase.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # If no neighbors, the simulation is over.
                break

            # Inner loop for processing scripts within a single timepoint.
            while True:
                # Wait for a script to arrive or for the timepoint to be marked as done.
                if self.device.script_arrived or self.device.timepoint_done.wait():
                    if self.device.script_arrived:
                        self.device.script_arrived = False
                        
                        # Add all received scripts to the thread pool's queue.
                        for (script, location) in self.device.scripts:
                            self.thread_pool.queue.put((neighbours, script, location))
                    else:
                        # Timepoint is done, clear the event and break the inner loop.
                        self.device.timepoint_done.clear()
                        self.device.script_arrived = True
                        break

            # Wait for the thread pool to finish all tasks for this timepoint.
            self.thread_pool.queue.join()

            # Synchronize with all other devices before proceeding to the next timepoint.
            self.device.barrier.wait()

        # Cleanly shut down the device's thread pool.
        self.thread_pool.end_threads()
