"""
This module simulates a distributed device network, where each device can process 
sensor data and communicate with its neighbors. It uses a thread pool to execute 
data processing scripts concurrently and a barrier for synchronization across devices.
"""

from threading import Event, Thread, Lock
from barrier import Barrier
from thread_pool import ThreadPool


class Device(object):
    """
    Represents a single device in the simulated network. Each device has a unique ID,
    manages its own sensor data, and executes assigned scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data, with locations as keys.
            supervisor (Supervisor): The supervisor object that manages the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.barrier = None
        self.scripts = []
        self.timepoint_done = Event()

        # A dictionary of locks to ensure thread-safe access to sensor data at different locations.
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
        Initializes the synchronization barrier for all devices in the network.
        This method is intended to be called by a single device (e.g., device 0)
        to set up a shared barrier for all devices.
        """
        if self.device_id == 0:
            barrier = Barrier(len(devices))
            self.barrier = barrier
            self.send_barrier(devices, barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        """Distributes the synchronization barrier to all other devices."""
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """Sets the synchronization barrier for this device."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script: The script to be executed.
            location: The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location in a thread-safe manner.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]

        return None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location and releases the lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """
        Shuts down the device by joining its associated thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device. This thread manages the device's lifecycle,
    including waiting for scripts, executing them using a thread pool, and synchronizing
    with other devices.
    """
    NR_THREADS = 8

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        self.thread_pool = ThreadPool(self.NR_THREADS)

    def run(self):
        """
        The main loop of the device thread. It continuously gets neighbours,
        processes scripts, and synchronizes with other devices.
        """
        self.thread_pool.set_device(self.device)

        while True:
            # Pre-condition: The device is ready to start a new timepoint simulation.
            # Invariant: Before this loop, the previous timepoint's computation is complete.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            while True:
                self.device.timepoint_done.wait()


                # Block Logic: This block processes all assigned scripts for the current timepoint.
                if self.device.scripts_available:
                    self.device.scripts_available = False

                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task((neighbours, location, script))
                else:
                    self.device.timepoint_done.clear()
                    self.device.scripts_available = True
                    break

            
            self.thread_pool.wait()

            # Synchronization point: Waits for all other devices to complete their computations for the current timepoint.
            self.device.barrier.wait()

        # Finalizes the thread pool, ensuring all tasks are completed.
        self.thread_pool.finish()


from threading import Thread
from Queue import Queue


class ThreadPool(object):
    """
    A simple thread pool implementation to execute tasks concurrently.
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
        """Associates a device with the thread pool."""
        self.device = device

    def submit_task(self, task):
        """Adds a task to the queue to be executed by a worker thread."""
        self.queue.put(task)

    def execute_task(self):
        """
        The target function for worker threads. Continuously fetches tasks
        from the queue and executes them.
        """

        while True:
            task = self.queue.get()

            neighbours = task[0]
            script = task[2]

            # A task with None script and neighbours is a signal to terminate the thread.
            if script is None and neighbours is None:
                self.queue.task_done()
                break

            self.run_script(task)
            self.queue.task_done()

    def run_script(self, task):
        """
        Executes a script using data from the device and its neighbors.
        """

        neighbours, location, script = task
        script_data = []

        
        # Gathers data from neighboring devices.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        
        # Gathers data from the current device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # The script is run with the collected data.
            result = script.run(script_data)

            
            # The result of the script is disseminated to the neighbors.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

            
            # The device also updates its own data with the result.
            self.device.set_data(location, result)

    def wait(self):
        """Blocks until all tasks in the queue have been processed."""
        self.queue.join()

    def finish(self):
        """
        Waits for all tasks to complete and then terminates all worker threads.
        """
        self.wait()

        
        # Sends termination signals to all worker threads.
        for _ in xrange(len(self.thread_list)):
            self.submit_task((None, None, None))

        
        # Waits for all worker threads to join.
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].join()