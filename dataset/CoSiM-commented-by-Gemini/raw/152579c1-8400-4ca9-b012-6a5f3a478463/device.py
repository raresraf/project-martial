"""
This module provides a device simulation framework using a thread pool
architecture, designed for a Python 2 environment.

Each device runs a single controller thread that dispatches tasks to a
dedicated thread pool. This version separates task orchestration from execution,
using a global barrier to synchronize devices at the end of each timepoint.

Note: This module depends on an external `barrier` module. It also contains
an unusual structure where `ThreadPool` is imported and also defined locally.
"""
from threading import Event, Thread, Lock
from barrier import Barrier
from thread_pool import ThreadPool


class Device(object):
    """
    Represents a device in the simulation.

    Each device has a single controller thread (`DeviceThread`) which manages a
    pool of worker threads for executing computational scripts. Synchronization
    between devices is handled by a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): The local sensor data for the device.
            supervisor: The supervisor object managing the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.barrier = None
        self.scripts = []
        self.timepoint_done = Event()

        
        self.locks = {}

        for location in sensor_data:
            self.locks[location] = Lock()

        
        self.scripts_available = False

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global barrier for all devices.

        The device with device_id 0 is responsible for creating and distributing
        the barrier to all other devices in the simulation.

        Args:
            devices (list): A list of all Device objects.
        """
        if self.device_id == 0:
            barrier = Barrier(len(devices))
            self.barrier = barrier
            self.send_barrier(devices, barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        """A static method to distribute the barrier to all devices."""
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """Sets the global barrier for this device."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A None script signals the end
        of assignments for a timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Acquires a lock and returns data for a given location.

        Warning: This method acquires a lock that is expected to be released
        by a subsequent call to `set_data`. This creates a fragile dependency
        between the two methods.

        Returns:
            The data at the location, or None if not found.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]

        return None

    def set_data(self, location, data):
        """
        Sets data for a given location and releases the corresponding lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """Shuts down the device by joining its controller thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    A controller thread for a single device.

    This thread's main role is to receive scripts, submit them as tasks to a
    thread pool, and synchronize with other devices using a global barrier.
    """
    NR_THREADS = 8

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        self.thread_pool = ThreadPool(self.NR_THREADS)

    def run(self):
        """The main execution loop for the device's controller thread."""
        self.thread_pool.set_device(self.device)

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            while True:
                self.device.timepoint_done.wait()


                if self.device.scripts_available:
                    self.device.scripts_available = False

                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task((neighbours, location, script))
                else:
                    self.device.timepoint_done.clear()
                    self.device.scripts_available = True
                    break

            
            self.thread_pool.wait()

            
            self.device.barrier.wait()

        
        self.thread_pool.finish()


from threading import Thread
from Queue import Queue


class ThreadPool(object):
    """
    A simple thread pool implementation for executing tasks from a queue.
    """
    def __init__(self, nr_threads):
        """
        Initializes the thread pool.

        Args:
            nr_threads (int): The number of worker threads to create.
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
        """Starts all worker threads."""
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].start()

    def set_device(self, device):
        """Sets a reference to the parent device."""
        self.device = device

    def submit_task(self, task):
        """Adds a task to the execution queue."""
        self.queue.put(task)

    def execute_task(self):
        """The target function for worker threads."""
        while True:
            task = self.queue.get()

            neighbours = task[0]
            script = task[2]

            if script is None and neighbours is None:
                self.queue.task_done()
                break

            self.run_script(task)
            self.queue.task_done()

    def run_script(self, task):
        """Executes a single script task."""
        neighbours, location, script = task
        script_data = []

        
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            
            result = script.run(script_data)

            
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

            
            self.device.set_data(location, result)

    def wait(self):
        """Blocks until all items in the queue have been processed."""
        self.queue.join()

    def finish(self):
        """Shuts down the thread pool."""
        self.wait()

        
        for _ in xrange(len(self.thread_list)):
            self.submit_task((None, None, None))

        
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].join()
