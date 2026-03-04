"""
Implements a thread pool-based simulation of a distributed device network.

This script defines a `ThreadPool` for managing concurrent script executions
and a `Device` class that uses this pool. The simulation uses a barrier for
synchronization and a per-location locking strategy, which requires careful
handling to avoid deadlocks.
"""

from Queue import Queue
from threading import Thread

class ThreadPool(object):

    """
    A generic thread pool for executing tasks concurrently.
    """

    def __init__(self, threads_count):
        """
        Initializes the thread pool.

        Args:
            threads_count: The number of worker threads to create.
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
        """Sets the parent device for context."""
        self.device = device

    def execute(self):
        """
        The main loop for a worker thread.

        Fetches tasks from the queue and executes them. A poison pill (None, None, None)
        is used to terminate the thread.
        """
        while True:

            neighbours, script, location = self.queue.get()

            if neighbours is None and script is None:
                self.queue.task_done()
                return

            self.run_script(neighbours, script, location)
            self.queue.task_done()

    def run_script(self, neighbours, script, location):
        """
        Executes a single script, gathering data and distributing the result.
        """
        script_data = []

        # Gather data from neighbors.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Gather data from self.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data)

            # Distribute result to neighbors.
            for device in neighbours:


                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

            # Set result for self.
            self.device.set_data(location, result)

    def submit(self, neighbours, script, location):
        """Submits a new task to the queue."""
        self.queue.put((neighbours, script, location))

    def wait_threads(self):
        """Blocks until all tasks in the queue are processed."""
        self.queue.join()

    def end_threads(self):
        """Shuts down the thread pool gracefully."""
        self.wait_threads()

        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)

        for thread in self.threads:
            thread.join()




from threading import Event, Thread, Lock

from barrier import Barrier
from ThreadPool import ThreadPool

class Device(object):
    """
    Represents a device in the simulation that uses a ThreadPool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id: Unique ID for the device.
            sensor_data: Local sensor data.
            supervisor: The simulation supervisor.
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
        Sets up the shared barrier, performed by device 0.
        """
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            self.send_barrier(devices, self.barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        """Distributes the shared barrier to other devices."""
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """Sets the shared barrier for this device."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_arrived = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safely retrieves data.

        Note: This method acquires a lock that is expected to be released by
        a subsequent call to set_data. This is a fragile design that can
        easily lead to deadlocks if not used carefully.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Thread-safely sets data and releases the lock acquired by get_data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread and its ThreadPool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        self.thread_pool = ThreadPool(8)

    def run(self):
        """
        The main loop for the device. It submits scripts to the thread pool
        and synchronizes at a barrier.
        """
        self.thread_pool.set_device(self.device)

        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            while True:

                
                if self.device.scripts_arrived or self.device.timepoint_done.wait():
                    if self.device.scripts_arrived:
                        self.device.scripts_arrived = False

                        
                        for (script, location) in self.device.scripts:
                            self.thread_pool.submit(neighbours, script, location)
                    else:
                        self.device.timepoint_done.clear()
                        self.device.scripts_arrived = True
                        break

            
            self.thread_pool.wait_threads()

            
            self.device.barrier.wait()

        
        self.thread_pool.end_threads()