"""
This module defines a distributed device simulation using a thread pool for
task management. It includes a Device class, a DeviceThread (the main control
thread), and a ThreadPool class for managing worker threads.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from threadpool import ThreadPool

class Device(object):
    """
    Represents a device in the simulation.

    Each device has a control thread (DeviceThread) that submits tasks to a
    thread pool. The device manages locks for its sensor data, but the locking
    mechanism is potentially problematic as get_data and set_data are not
    symmetrically locking and unlocking.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: A dictionary of the device's sensor data.
            supervisor: The supervisor managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        # A dictionary of locks for each sensor data location.
        self.locations_locks = {location: Lock() for location in sensor_data}

        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for all devices, coordinated by device 0.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device. If script is None, signals the end
        of the timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Gets data from a location. WARNING: This method acquires a lock that
        is not released here. The caller is responsible for releasing it via
        set_data, which is a bug-prone design.
        """
        if location in self.sensor_data:
            self.locations_locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Sets data at a location and releases the lock acquired by get_data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations_locks[location].release()

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    It manages a ThreadPool to which it submits script execution tasks.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The Device this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(8, device)

    def run(self):
        """The main execution loop for the control thread."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            while True:
                # Wait for the end of the timepoint, ensuring scripts are processed.
                if self.device.timepoint_done.wait() and not self.device.script_received.is_set():
                    self.device.timepoint_done.clear()
                    self.device.script_received.set()
                    break

                if self.device.script_received.is_set():
                    self.device.script_received.clear()
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task(script, location, neighbours)

            # Wait for all tasks in the current timepoint to be completed.
            self.thread_pool.tasks_queue.join()

            # Wait at the barrier for all other devices to finish.
            self.device.barrier.wait()

        self.thread_pool.join_threads()


from Queue import Queue

class ThreadPool(object):
    """
    A simple thread pool for executing tasks concurrently.
    """

    def __init__(self, number_threads, device):
        """
        Initializes the ThreadPool.

        Args:
            number_threads: The number of worker threads in the pool.
            device: The device this thread pool belongs to.
        """
        self.number_threads = number_threads
        self.device_threads = []
        self.device = device
        self.tasks_queue = Queue(number_threads)

        for _ in range(number_threads):
            thread = Thread(target=self.apply_scripts)
            self.device_threads.append(thread)
            thread.start()

    def apply_scripts(self):
        """The target function for the worker threads."""
        while True:
            script, location, neighbours = self.tasks_queue.get()

            if neighbours is None and script is None:
                self.tasks_queue.task_done()
                return

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

            if script_data:
                result = script.run(script_data)

                # Update data on neighbors.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                # Update data on the current device.
                self.device.set_data(location, result)

            self.tasks_queue.task_done()

    def submit_task(self, script, location, neighbours):
        """Submits a task to the thread pool's queue."""
        self.tasks_queue.put((script, location, neighbours))

    def join_threads(self):
        """Joins all threads in the pool."""
        self.tasks_queue.join()

        for _ in range(len(self.device_threads)):
            self.submit_task(None, None, None)

        for thread in self.device_threads:
            thread.join()
