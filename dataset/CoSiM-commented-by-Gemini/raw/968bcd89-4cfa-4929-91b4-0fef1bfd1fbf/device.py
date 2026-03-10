
"""
Models a distributed network of devices that process sensor data concurrently.

This script simulates a system of interconnected devices, each running in its own
thread and utilizing a thread pool to execute data processing scripts. It uses
synchronization primitives like Barriers and Locks to coordinate data access and
ensure consistent state across devices at discrete timepoints.
"""

from threading import Event, Thread, Lock
from barrier import Barrier
from thread_pool import ThreadPool

class Device(object):
    """Represents a single device in the distributed sensor network.

    Each device manages its own sensor data, executes assigned scripts, and
    communicates with neighboring devices. It uses a dedicated thread
    (DeviceThread) to manage its lifecycle and a ThreadPool for script execution.

    Attributes:
        device_id (int): A unique identifier for the device.
        num_threads (int): The number of worker threads in the device's thread pool.
        sensor_data (dict): A dictionary holding the device's sensor readings,
                            keyed by location.
        supervisor (Supervisor): An object responsible for providing neighborhood
                                 information.
        scripts (list): A list of (script, location) tuples to be executed.
        timepoint_done (Event): An event that signals the completion of a
                                simulation timepoint.
        barrier (Barrier): A synchronization primitive to ensure all devices
                           complete a timepoint before starting the next.
        location_lock (dict): A dictionary of locks, keyed by location, to ensure
                              thread-safe access to sensor data.
        thread (DeviceThread): The main thread of execution for this device.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.num_threads = 8
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.location_lock = {}
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes shared synchronization objects for a group of devices.

        This method is typically called by a master device (device_id == 0) to
        set up a shared barrier for all devices and create a global set of locks
        for all sensor locations.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            if self.barrier is None:
                self.barrier = Barrier(len(devices))

            for device in devices:
                for location in device.sensor_data:
                    if location not in self.location_lock:
                        self.location_lock[location] = Lock()
                device.barrier = self.barrier
                device.location_lock = self.location_lock

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device.

        A script value of None is a sentinel to signal the end of a timepoint.

        Args:
            script (Script): The script object to execute.
            location (str): The location context for the script execution.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location.

        Args:
            location (str): The location from which to retrieve data.

        Returns:
            The sensor data if the location exists, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location.

        Args:
            location (str): The location to update.
            data: The new sensor data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's execution thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main execution thread for a Device.

    This thread manages the device's lifecycle, orchestrating script execution
    and synchronization between timepoints.

    Attributes:
        device (Device): The device instance this thread belongs to.
        thread_pool (ThreadPool): The pool of worker threads for executing scripts.
    """
    
    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(device, self.device.num_threads)

    def run(self):
        """The main loop for the device thread.

        It continuously waits for a timepoint to be marked as done, queues all
        assigned scripts for execution in the thread pool, waits for their
        completion, and then synchronizes with other devices using a barrier.
        The loop terminates when the supervisor signals no more neighbors.
        """
        self.thread_pool.start_threads()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            for script, location in self.device.scripts:
                self.thread_pool.queue.put((script, location, neighbours))

            
            
            self.thread_pool.queue.join()
            self.device.barrier.wait()
            self.device.timepoint_done.clear()

        
        for _ in range(self.device.num_threads):
            self.thread_pool.queue.put((None, None, None))

        self.thread_pool.end_threads()


from Queue import Queue
from threading import Thread

class ThreadPool(object):
    """A simple thread pool for concurrent script execution.

    Attributes:
        queue (Queue): A queue to hold tasks for worker threads.
        device (Device): The parent device.
        threads (list): A list of worker Thread objects.
        num_threads (int): The number of threads in the pool.
    """
    
    def __init__(self, device, num_threads):
        """Initializes the ThreadPool."""
        self.queue = Queue(num_threads)
        self.device = device
        self.threads = []
        self.num_threads = num_threads

    def start_threads(self):
        """Creates and starts all worker threads in the pool."""
        for _ in range(self.num_threads):
            self.threads.append(Thread(target=self.run))



        for thread in self.threads:
            thread.start()

    def run(self):
        """The target function for worker threads.

        Continuously fetches tasks (script, location, neighbours) from the
        queue and executes them. A task of (None, None, None) is a sentinel
        value to terminate the thread.
        """
        while True:
            script, location, neighbours = self.queue.get()

            if script is None and location is None:
                return

            self.run_script(script, location, neighbours)
            self.queue.task_done()

    def run_script(self, script, location, neighbours):
        """Executes a single script.

        This function acquires a lock for the specified location to ensure data
        consistency. It gathers data from the current device and its neighbors,
        runs the script with the collected data, and then propagates the result
        back to the same set of devices.

        Args:
            script (Script): The script to execute.
            location (str): The location context for the script.
            neighbours (list): A list of neighboring Device objects.
        """
        with self.device.location_lock[location]:
            script_data = []

            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(.
        for thread in self.threads:
            thread.join()
