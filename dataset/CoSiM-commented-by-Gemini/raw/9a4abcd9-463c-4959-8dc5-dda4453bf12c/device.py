"""
Models a distributed network of devices that process sensor data concurrently.

This script implements a device simulation using a WorkerFactory to manage a
pool of threads for script execution.

NOTE: This implementation contains several severe flaws. The locking mechanism
is fundamentally broken: locks are not shared between devices, and they are
acquired in `get_data` but released in `set_data`, which will lead to deadlocks
and race conditions. The control flow in `DeviceThread` is also highly complex
and difficult to reason about.
"""

from threading import Event, Thread, Lock
from barrier import Barrier
from workerfactory import WorkerFactory

class Device(object):
    """Represents a single device in the distributed sensor network.

    This implementation uses a barrier for inter-device synchronization but has
    a flawed, non-shared locking mechanism for data access.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor readings.
        supervisor (Supervisor): An object for retrieving neighbor information.
        thread (DeviceThread): The main orchestration thread for this device.
        locks (list): A list of (location, Lock) tuples. Flawed: these locks
                      are not shared between devices.
        barrier (Barrier): A shared barrier for timepoint synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.locks = []
        self.barrier = None

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes the shared barrier and this device's local locks."""
        num_devices = len(devices)
        if self.barrier is None and self.device_id == 0:
            self.barrier = Barrier(num_devices)
            for dev in devices:
                if dev.barrier is None:
                    dev.barrier = self.barrier
        for loc in self.sensor_data:
            self.locks.append((loc, Lock()));

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data, but contains a flawed locking mechanism.

        BUG: This method acquires a lock and does not release it, holding it
        until `set_data` is called, which is a pattern that leads to deadlock.
        """
        if location in self.sensor_data:
            for iter in self.locks:
                if iter[0] == location:
                    iter[1].acquire();
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Updates sensor data and releases a lock acquired by `get_data`.

        BUG: This method's release corresponds to an acquisition in a
        completely different function, which is a dangerous and broken pattern.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            for iter in self.locks:
                if iter[0] == location:
                    iter[1].release();

    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main execution thread for a Device.

    This thread uses a `WorkerFactory` to manage a pool of eight worker threads.
    Its main loop's control flow is convoluted and likely buggy.
    """
    num_cores = 8
    def __init__(self, device):
        """Initializes the DeviceThread and its WorkerFactory."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.worker_factory = WorkerFactory(DeviceThread.num_cores, device)

    def run(self):
        """The main control loop for the device.

        Its logic for waiting on events and dispatching scripts is complex and
        hard to reason about.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            while True:
                if self.device.timepoint_done.wait():
                    if self.device.script_received.isSet():
                        self.device.script_received.clear()
                        for (script, location) in self.device.scripts:
                            self.worker_factory.add_tasks((neighbours, script, location))
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break

            self.worker_factory.wait_for_finish()
            self.device.barrier.wait()
        self.worker_factory.shutdown()

from Queue import Queue
from threading import Thread

class WorkerFactory(object):
    """A factory to create and manage a pool of worker threads.

    This class encapsulates a standard thread pool pattern using a task queue.
    """
    def __init__(self, num_workers, parent_device):
        """Initializes the thread pool."""
        self.num_workers = num_workers
        self.task_queue = Queue(num_workers)
        self.worker_threads = []
        self.current_device = parent_device
        self.start_workers()

    def start_workers(self):
        """Creates and starts all worker threads."""
        for _ in range(0, self.num_workers):
            worker_thread = Worker(self.task_queue, self.current_device)
            self.worker_threads.append(worker_thread)
        for worker in self.worker_threads:
            worker.start()

    def add_tasks(self, necessary_data):
        """Adds a task to the worker queue."""
        self.task_queue.put(necessary_data)

    def wait_for_finish(self):
        """Blocks until all tasks in the queue are processed."""
        self.task_queue.join()

    def shutdown(self):
        """Shuts down all worker threads gracefully."""
        self.task_queue.join()
        for _ in xrange(self.num_workers):
            self.add_tasks((None, None, None))

        for worker in self.worker_threads:
            worker.join()

class Worker(Thread):
    """A worker thread that executes tasks from a queue."""
    def __init__(self, task_queue, parent_device):
        """Initializes the worker."""
        Thread.__init__(self)
        self.my_queue = task_queue
        self.current_device = parent_device

    def run(self):
        """The main loop for the worker thread.

        It continuously fetches tasks from the queue and processes them until a
        shutdown signal (None) is received.
        """
        while True:
            neigh, script, location = self.my_queue.get()
            if neigh is None or script is None or location is None:
                self.my_queue.task_done()
                break

            script_data = []
            
            # This data collection step will likely lead to deadlock due to the
            # flawed locking in the Device class.
            for device in neigh:
                if self.current_device.device_id != device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

            data = self.current_device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)

                for device in neigh:
                    if self.current_device.device_id != device.device_id:
                        device.set_data(location, result)
                
                self.current_device.set_data(location, result)
            self.my_queue.task_done()
