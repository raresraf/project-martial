"""
@file MyThread.py
@brief A distributed device simulation using a custom thread pool and an
       asymmetric locking protocol.

This script is split into two main parts.
1. `MyThread`: A reusable, queue-based thread pool for executing tasks.
2. `Device` and `DeviceThread`: A simulation of a distributed network where each
   device uses an instance of `MyThread` for parallel processing.

The simulation uses a global barrier for time-step synchronization and a highly
dangerous asymmetric locking pattern for data access, where one method acquires a
lock and another is expected to release it.
"""

from Queue import Queue
from threading import Thread

class MyThread(object):
    """
    A custom, queue-based thread pool for executing tasks in parallel.
    """
    def __init__(self, threads_count):
        """
        Initializes the thread pool and starts the worker threads.
        @param threads_count The number of worker threads to create.
        """
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None # Reference to the parent device, to be injected later.

        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)
            new_thread.start()

    def execute(self):
        """The main loop for a worker thread; gets and executes tasks from the queue."""
        while True:
            # Block until a task is available.
            neighbours, script, location = self.queue.get()

            # A `None` tuple is a "poison pill" to terminate the thread.
            if neighbours is None and script is None:
                self.queue.task_done()
                return

            self.run_script(neighbours, script, location)
            self.queue.task_done()

    def run_script(self, neighbours, script, location):
        """
        Executes a single script.
        @warning This method participates in a dangerous, deadlock-prone locking
        protocol by calling the asymmetric `get_data` (lock) and `set_data` (unlock)
        methods on multiple devices.
        """
        script_data = []

        # Block Logic: Acquire locks and get data from neighbors and self.
        # This sequence of acquiring multiple locks is highly prone to deadlock.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is None:
                    continue
                script_data.append(data)

        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = script.run(script_data)

            # Block Logic: Set data and release locks on neighbors and self.
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    continue
                device.set_data(location, result)
            self.device.set_data(location, result)

    def end_threads(self):
        """Gracefully shuts down the thread pool."""
        # Wait for all tasks in the queue to be completed.
        self.queue.join()
        # Send a "poison pill" for each thread to signal termination.
        for _ in xrange(len(self.threads)):
            self.queue.put((None, None, None))
        # Wait for all worker threads to finish.
        for thread in self.threads:
            thread.join()


from threading import Event, Thread, Lock
from barrier import Barrier
# MyThread is in the same file, but this shows dependency.
# from MyThread import MyThread

class Device(object):
    """
    Represents a node in the network, managing its data, locks, and control thread.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.new_adds()
        self.thread.start()

    def new_adds(self):
        """Helper method for initializing device properties."""
        self.barrier = None
        self.locations = {location: Lock() for location in self.sensor_data}
        self.script_arrived = False

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes the shared barrier, orchestrated by device 0."""
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to the device for the current timepoint."""
        self.set_boolean(script)
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def set_boolean(self, script):
        """Helper to track if scripts have arrived."""
        if script is not None:
            self.script_arrived = True

    def acquire_location(self, location):
        """Acquires the lock for a specific location on this device."""
        if location in self.sensor_data:
            self.locations[location].acquire()

    def get_data(self, location):
        """
        Retrieves data for a location AFTER acquiring its lock.
        @warning Acquires a lock and does not release it. The caller must ensure
        a corresponding `set_data` call is made to release the lock.
        """
        self.acquire_location(location)
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates data for a location AND releases its lock.
        @warning Releases a lock assumed to be held from a prior `get_data` call.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations[location].release()

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()

class DeviceThread(Thread):
    """The main control thread for a device, which manages a `MyThread` thread pool."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = MyThread(8)

    def run(self):
        """Main time-stepping loop."""
        # Inject device reference into the thread pool.
        self.thread_pool.device = self.device

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal

            # This loop waits for all scripts to be assigned, then submits them.
            while True:
                if self.device.script_arrived or self.device.timepoint_done.wait():
                    if self.device.script_arrived:
                        self.device.script_arrived = False
                        for (script, location) in self.device.scripts:
                            self.thread_pool.queue.put((neighbours, script, location))
                    else: # timepoint_done was set
                        self.device.timepoint_done.clear()
                        self.device.script_arrived = True # Reset for next cycle
                        break

            # Wait for all submitted tasks for this timepoint to be completed.
            self.thread_pool.queue.join()

            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()

        self.thread_pool.end_threads()
