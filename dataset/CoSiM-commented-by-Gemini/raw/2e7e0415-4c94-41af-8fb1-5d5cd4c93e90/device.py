"""
@file device.py
@brief A distributed device simulation using a helper class to manage a worker thread pool.
@details This module implements a producer-consumer pattern where a master `DeviceThread`
dispatches tasks to a pool of worker threads. The thread pool logic is encapsulated
in a `ThreadCollection` class. This version correctly uses `queue.join()` and `task_done()`
for synchronization but contains a hazardous locking pattern in its data access methods.
The file also includes its own class definitions despite importing them.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue
from utility import ReusableBarrierCond, ThreadCollection

class Device(object):
    """
    @brief Represents a single device, which owns a master thread and a worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        # A dictionary mapping data locations to Lock objects.
        self.locks = {}
        # An event to signal that the initial setup is complete.
        self.setup = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """@brief Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Initializes and distributes shared resources (barrier and locks).
        @details The device with ID 0 acts as the master, creating the shared resources
        and signaling other devices to adopt them once ready.
        """
        for location in self.sensor_data:
            self.locks[location] = Lock()
        
        # Master device (ID 0) creates the shared barrier.
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                device.barrier = barrier
            # Signal all other devices that setup is complete.
            for device in devices:
                device.setup.set()

    def assign_script(self, script, location):
        """@brief Adds a script to the list or signals that the timepoint is done."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Acquires a lock and retrieves sensor data.
        @warning DANGEROUS LOCKING PATTERN: This method acquires a lock but does not release it.
        It relies on a future call to `set_data` for the same location to release the lock.
        This can easily lead to deadlocks if `set_data` is not called, or is called on a
        different location, or by a different thread that doesn't own the lock.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        @brief Updates sensor data and releases the lock for that location.
        @warning This method releases a lock it did not acquire, completing the dangerous pattern.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """@brief Shuts down the device by joining its main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief The master thread for a device. It produces tasks for the worker pool.
    @details This implementation correctly synchronizes with its worker pool by using `queue.join()`.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.worker_threads = ThreadCollection(self.device, 8)

    def run(self):
        """@brief Main execution loop."""
        self.device.setup.wait() # Wait for shared resources to be initialized.

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal.

            self.device.timepoint_done.wait() # Wait for all scripts to be assigned.

            # Add all assigned scripts as tasks to the worker queue.
            for (script, location) in self.device.scripts:
                self.worker_threads.add_task(script, location, neighbours)

            self.device.timepoint_done.clear()
            
            # Correctly waits for all tasks in the queue to be processed by workers.
            self.worker_threads.queue.join()
            
            # After all work is done, synchronize with other devices.
            self.device.barrier.wait()

        self.worker_threads.end_workers() # Cleanly shut down the worker pool.


# ======================================================================================
# The following are helper classes, likely intended to be in a separate 'utility' module.
# ======================================================================================

class ThreadCollection(object):
    """
    @brief A helper class that encapsulates a thread pool and a task queue.
    """
    def __init__(self, device, num_threads):
        self.device = device
        self.threads = []
        self.queue = Queue(num_threads)
        self.create_workers(num_threads)
        self.start_workers()

    def __str__(self):
        return "Thread collection belonging to device %d" % self.device.device_id

    def create_workers(self, num_threads):
        """@brief Creates the worker threads for the pool."""
        for _ in range(num_threads): # xrange is for Python 2; range is standard in Python 3.
            new_thread = Thread(target=self.run_tasks)
            self.threads.append(new_thread)

    def start_workers(self):
        """@brief Starts all worker threads in the pool."""
        for thread in self.threads:
            thread.start()

    def run_tasks(self):
        """@brief The target function for each worker thread; consumes tasks from the queue."""
        while True:
            (neighbours, script, location) = self.queue.get()
            
            # A 'None' task is a "poison pill" to signal termination.
            if location is None and neighbours is None and script is None:
                self.queue.task_done()
                break

            self.run_script(script, location, neighbours)
            self.queue.task_done() # Correctly signals that the task is finished.

    def run_script(self, script, location, neighbours):
        """
        @brief Executes a single script.
        @note This method's correctness is compromised by the flawed locking in get_data/set_data.
        """
        script_data = []
        # Gather data from neighbors.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        
        # Gather data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = script.run(script_data)
            # Broadcast the result to neighbors and local device.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)
            self.device.set_data(location, result)

    def add_task(self, script, location, neighbours):
        """@brief Adds a new task to the worker queue."""
        self.queue.put((neighbours, script, location))

    def end_workers(self):
        """@brief Shuts down the thread pool cleanly."""
        self.queue.join()
        for _ in range(len(self.threads)): # xrange is for Python 2.
            self.add_task(None, None, None)
        for thread in self.threads:
            thread.join()

class ReusableBarrierCond(object):
    """
    @brief A barrier implementation using a `threading.Condition`.
    @warning This is not a safe reusable barrier. It is vulnerable to a race condition
    if threads loop and call wait() again before all threads have woken up from the
    previous wait cycle.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
