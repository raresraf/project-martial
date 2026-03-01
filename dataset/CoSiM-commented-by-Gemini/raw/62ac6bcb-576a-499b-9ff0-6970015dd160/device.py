"""
@file device.py
@brief A distributed device simulation using a custom thread pool and an
       asymmetric locking protocol.

This script models a network of devices operating in synchronized time steps.
Each device's main thread delegates parallel work to a custom, queue-based
thread pool. The most notable feature is its dangerous locking mechanism where
`get_data` acquires a lock and `set_data` releases it, creating a high risk of
deadlocks between communicating devices.
"""

from threading import Event, Thread, Lock

from barrier import ReusableBarrierCond
from threadpool import ThreadPool

class Device(object):
    """
    Represents a node in the distributed network. It owns its data and the locks
    that protect it.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        # A dictionary of locks, one for each location this device owns.
        self.locations_locks = dict([(location, Lock()) for location in sensor_data])
        
        # A shared barrier, to be provided by the central setup.
        self.barrier = None
        
        # The main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes the shared barrier, orchestrated by device 0."""
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to the device for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location after acquiring its lock.
        @warning This method acquires a lock and DOES NOT release it. The caller
        is responsible for ensuring a corresponding `set_data` call is made to
        release the lock.
        """
        if location in self.sensor_data:
            self.locations_locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location and releases its lock.
        @warning This method releases a lock that is assumed to have been
        acquired by a previous call to `get_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations_locks[location].release()

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, managing a ThreadPool to execute scripts.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(8, device)

    def run(self):
        """Main execution loop, processing timepoints sequentially."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal

            # This inner loop's event logic is complex, but aims to wait until
            # all scripts for the timepoint are received.
            while True:
                if self.device.timepoint_done.wait() and not self.device.script_received.is_set():
                    self.device.timepoint_done.clear()
                    self.device.script_received.set()
                    break

                if self.device.script_received.is_set():
                    self.device.script_received.clear()
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task(script, location, neighbours)

            # Block Logic: Correctly wait for all tasks submitted to the thread pool
            # for this timepoint to complete before proceeding.
            self.thread_pool.tasks_queue.join()

            # Wait at the global barrier to synchronize with all other devices.
            self.device.barrier.wait()

        # Cleanly shut down the thread pool upon termination.
        self.thread_pool.join_threads()


from threading import Thread
from Queue import Queue

class ThreadPool(object):
    """A custom, queue-based thread pool for executing script tasks."""
    def __init__(self, number_threads, device):
        self.number_threads = number_threads
        self.device_threads = []
        self.device = device
        self.tasks_queue = Queue(number_threads)

        for _ in xrange(0, number_threads):
            thread = Thread(target=self.apply_scripts)
            self.device_threads.append(thread)
            thread.start()

    def apply_scripts(self):
        """
        The main execution loop for a worker thread. It gets a task from the
        queue and executes it.
        """
        while True:
            script, location, neighbours = self.tasks_queue.get()

            # A `None` task is a "poison pill" to terminate the thread.
            if neighbours is None and script is None:
                self.tasks_queue.task_done()
                return

            script_data = []
            
            # Block Logic: Acquire locks and get data from neighbors.
            # This sequence of acquiring multiple locks is highly prone to deadlock.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Acquire lock and get data from self.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)

                # Block Logic: Set data and release locks on neighbors and self.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                self.device.set_data(location, result)

            self.tasks_queue.task_done()

    def submit_task(self, script, location, neighbours):
        """Adds a new script execution task to the queue."""
        self.tasks_queue.put((script, location, neighbours))

    def join_threads(self):
        """Gracefully shuts down the thread pool."""
        # Wait for all tasks in the queue to be completed.
        self.tasks_queue.join()

        # Send a "poison pill" for each thread to signal termination.
        for _ in xrange(0, len(self.device_threads)):
            self.submit_task(None, None, None)

        # Wait for all worker threads to finish.
        for thread in self.device_threads:
            thread.join()
