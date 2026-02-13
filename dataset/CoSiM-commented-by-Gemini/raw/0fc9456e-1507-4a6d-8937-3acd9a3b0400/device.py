"""
This module simulates a distributed system of devices using a thread pool
pattern for concurrent script execution.

The architecture is composed of several classes:
- Worker/WorkerFactory: A standard thread pool implementation where a factory
  manages a queue of tasks and a set of worker threads that execute them.
- Device: Represents a node in the system. It receives scripts from a supervisor
  and uses its own control thread (`DeviceThread`) to manage its lifecycle.
- DeviceThread: The main control thread for a Device. It receives tasks and
  dispatches them to a `WorkerFactory`.

Warning: This implementation contains a fragile and potentially dangerous locking
mechanism that can easily lead to deadlocks.
"""

from threading import Event, Thread, Lock
from Queue import Queue
from barrier import Barrier
from workerfactory import WorkerFactory

class Device(object):
    """
    Represents a single device in the distributed simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): The device's local sensor data.
            supervisor (Supervisor): The central controller of the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # A list of (location, Lock) tuples. This is an inefficient structure.
        self.locks = []
        # The global synchronization barrier.
        self.barrier = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (barrier, locks).
        
        Args:
            devices (list): A list of all devices in the simulation.
        """
        num_devices = len(devices)
        # Device 0 is the coordinator and creates the shared barrier.
        if self.barrier is None and self.device_id == 0:
            self.barrier = Barrier(num_devices)
            for dev in devices:
                if dev.barrier is None:
                    dev.barrier = self.barrier
        # Each device creates its own list of locks only for the locations it knows about.
        for loc in self.sensor_data:
            self.locks.append((loc, Lock()));

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A 'None' script signals the end of script assignments.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data for a given location and acquires a lock.

        @warning Asymmetric Locking: This function acquires a lock but does not
        release it. The lock is expected to be released later by `set_data`.
        This is a highly fragile pattern. If `set_data` is not called for any
        reason after `get_data`, the lock will be held forever, causing a deadlock.
        """
        if location in self.sensor_data:
            # Inefficiently searches for the lock in a list.
            for iter in self.locks:
                if iter[0] == location:
                    iter[1].acquire();
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Updates data for a given location and releases a lock.

        @warning Asymmetric Locking: This function releases a lock that it did
        not acquire. This is part of a fragile pattern with `get_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Inefficiently searches for the lock to release it.
            for iter in self.locks:
                if iter[0] == location:
                    iter[1].release();

    def shutdown(self):
        """Shuts down the main device thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device, responsible for orchestrating script
    execution and synchronization.
    """
    num_cores = 8
    def __init__(self, device):
        """
        Initializes the DeviceThread.
        
        Args:
            device (Device): The parent device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        # Creates a thread pool for executing tasks.
        self.worker_factory = WorkerFactory(DeviceThread.num_cores, device)

    def run(self):
        """The main execution loop of the device thread."""
        while True:
            # Get neighbors for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.
            
            # This loop waits for scripts and dispatches them.
            while True:
                # Wait for the supervisor to signal that all scripts are assigned.
                if self.device.timepoint_done.wait():
                    # Check if any scripts were actually received.
                    if self.device.script_received.isSet():
                        self.device.script_received.clear()
                        for (script, location) in self.device.scripts:
                            self.worker_factory.add_tasks((neighbours, script, location))
                    else:
                        # No scripts received, end the timepoint processing.
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break

            
            # Wait for all dispatched tasks to be completed by the worker threads.
            self.worker_factory.wait_for_finish()

            
            # Synchronize with all other devices at the end of the timepoint.
            self.device.barrier.wait()

        
        # Cleanly shut down the worker thread pool.
        self.worker_factory.shutdown()

class WorkerFactory(object):
    """
    A factory class for creating and managing a pool of worker threads.
    """
    def __init__(self, num_workers, parent_device):
        """
        Initializes the WorkerFactory.
        
        Args:
            num_workers (int): The number of worker threads to create.
            parent_device (Device): The device that owns this factory.
        """
        self.num_workers = num_workers
        self.task_queue = Queue(num_workers)
        self.worker_threads = []
        self.current_device = parent_device
        self.start_workers()

    def start_workers(self):
        """Creates and starts the worker threads."""
        for _ in range(0, self.num_workers):
            worker_thread = Worker(self.task_queue, self.current_device)
            self.worker_threads.append(worker_thread)
        for worker in self.worker_threads:
            worker.start()

    def add_tasks(self, necessary_data):
        """Adds a new task to the work queue."""
        self.task_queue.put(necessary_data)

    def wait_for_finish(self):
        """Blocks until all tasks in the queue are processed."""
        self.task_queue.join()

    def shutdown(self):
        """Shuts down all worker threads gracefully."""
        self.task_queue.join()
        # Send a termination signal to each worker.
        for _ in xrange(self.num_workers):
            self.add_tasks((None, None, None))

        for worker in self.worker_threads:
            worker.join()

class Worker(Thread):
    """
    A worker thread that processes tasks from a queue.
    """
    def __init__(self, task_queue, parent_device):
        Thread.__init__(self)
        self.my_queue = task_queue
        self.current_device = parent_device

    def run(self):
        """The main execution loop for the worker."""
        while True:
            # Get a task from the queue.
            neigh, script, location = self.my_queue.get()
            # A 'None' task is the termination signal.
            if neigh is None or script is None or location is None:
                self.my_queue.task_done()
                break

            script_data = []
            
            # Gather data from neighbors. This triggers the fragile locking in Device.get_data().
            for device in neigh:
                if self.current_device.device_id != device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

            
            data = self.current_device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Execute the script.
                result = script.run(script_data)

                # Broadcast the results. This triggers the lock release in Device.set_data().
                for device in neigh:
                    if self.current_device.device_id != device.device_id:
                        device.set_data(location, result)
                
                self.current_device.set_data(location, result)
            self.my_queue.task_done()
