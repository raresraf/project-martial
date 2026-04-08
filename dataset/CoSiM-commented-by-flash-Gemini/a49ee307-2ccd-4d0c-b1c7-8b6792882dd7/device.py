"""
This module provides a simulation of a distributed system of devices.

It features an efficient, thread-pool-based architecture. A master thread for
each device dispatches script execution tasks to a persistent pool of worker
threads. Shared state and synchronization primitives are cleanly encapsulated in
a dedicated `SharedDeviceData` object.
"""

from threading import Thread, Lock
from Queue import Queue

# The following classes are assumed to be in a 'utils' module.
# from utils import SharedDeviceData
# from utils import ThreadPool

class Device(object):
    """
    Represents a single device in the simulation.

    Its primary role is to receive scripts and enqueue them for its master
    thread (`DeviceThread`).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): A dictionary of local sensor data.
            supervisor: The supervisor object providing network topology.
        """
        self.device_id = device_id
        self.num_cores = 8  # Number of worker threads in the thread pool.
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.new_scripts = Queue()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes the shared data object to all devices.

        Device 0 is responsible for creating the single `SharedDeviceData`
        instance that all other devices will use for synchronization.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        if self.device_id == 0:
            # Create the central object for shared state.
            shared_data = SharedDeviceData(len(devices))
            
            # Pre-populate locks for existing sensor data locations.
            for data in self.sensor_data:
                if data not in shared_data.location_locks:
                    shared_data.location_locks[data] = Lock()

            # Assign the shared object to all devices.
            for dev in devices:
                dev.shared_data = shared_data

    def assign_script(self, script, location):
        """
        Assigns a script to the device by putting it on a queue.

        Args:
            script: The script to execute, or None to signal end of timepoint.
            location (int): The data location for the script.
        """
        self.new_scripts.put((script, location))

    def get_data(self, location):
        """Gets data from the device's local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets data in the device's local sensor data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the master device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The master thread for a device, orchestrating a thread pool.

    It pulls newly assigned scripts from a queue, submits them as tasks to a
    `ThreadPool`, and synchronizes with other devices at the end of each
    timepoint.
    """

    def __init__(self, device):
        """
        Initializes the master DeviceThread.

        Args:
            device (Device): The parent device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main execution loop for the master thread."""
        thread_pool = ThreadPool(self.device.num_cores)
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Supervisor signals shutdown.
                break

            # Re-submit scripts from the previous timepoint.
            for (script, location) in self.device.scripts:
                thread_pool.submit(RunScript(script, location, neighbours,
                                             self.device))

            # Process new scripts for the current timepoint.
            while True:
                (script, location) = self.device.new_scripts.get()
                if script is None: # End of timepoint signal.
                    break

                # Lazily initialize a lock for a new location if needed.
                with self.device.shared_data.ll_lock:
                    if location not in self.device.shared_data.location_locks:
                        self.device.shared_data.location_locks[location] = Lock()

                # Submit the script as a task to the thread pool.
                thread_pool.submit(RunScript(script, location, neighbours,
                                             self.device))
                self.device.scripts.append((script, location))

            thread_pool.shutdown() # Prevent new tasks from being added.
            thread_pool.wait_termination(False) # Wait for existing tasks to finish.

            # Synchronize with all other master threads before the next timepoint.
            self.device.shared_data.timepoint_barrier.wait()

        thread_pool.wait_termination() # Fully shut down the thread pool.

class RunScript(object):
    """
    A callable task object representing a single script execution.

    This encapsulates the logic for running one script, which allows it to be
    submitted to a thread pool.
    """

    def __init__(self, script, location, neighbours, device):
        """
        Initializes the script task.

        Args:
            script: The script to execute.
            location (int): The data location to operate on.
            neighbours (list): A list of neighboring devices.
            device (Device): The parent device of the thread executing this task.
        """
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """The core logic for executing the script."""
        # Get the specific lock for this location from the shared data object.
        with self.device.shared_data.ll_lock:
            lock = self.device.shared_data.location_locks[self.location]

        script_data = []
        with lock:
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            # Gather local data.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Run the script and update data on all relevant devices.
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)

# --- Start of 'utils' module classes ---

from threading import Condition, Semaphore, Event, Lock
from threading import Thread
from Queue import Queue

class CyclicBarrier(object):
    """A simple, reusable barrier implementation using a Condition variable."""

    def __init__(self, parties):
        """
        Initializes the CyclicBarrier.

        Args:
            parties (int): The number of threads to wait for.
        """
        self.parties = parties
        self.count = 0
        self.condition = Condition()

    def wait(self):
        """Waits until all parties have called wait() on this barrier."""
        with self.condition:
            self.count += 1
            if self.count == self.parties:
                self.condition.notifyAll() # Release all waiting threads.
                self.count = 0  # Reset for reuse.
            else:
                self.condition.wait()

class ThreadPool(object):
    """A pool of worker threads that execute tasks from a queue."""

    def __init__(self, num_threads):
        """
        Initializes the ThreadPool.

        Args:
            num_threads (int): The number of persistent worker threads.
        """
        self.num_threads = num_threads
        self.task_queue = Queue() 
        self.num_tasks = Semaphore(0) # Counts pending tasks.
        self.stop_signal = Event() # Signals workers to terminate permanently.
        self.shutdown_signal = Event() # Signals the pool to stop accepting new tasks.

        self.threads = []
        for i in xrange(0, num_threads):
            self.threads.append(Worker(self.task_queue,
                                       self.num_tasks,
                                       self.stop_signal))

        for i in xrange(0, num_threads):
            self.threads[i].start()

    def submit(self, task):
        """
        Submits a task to the thread pool for execution.

        Args:
            task: A callable object with a `run()` method.
        """
        if self.shutdown_signal.is_set():
            return 

        self.task_queue.put(task)
        self.num_tasks.release()

    def shutdown(self):
        """Prevents new tasks from being submitted to the pool."""
        self.shutdown_signal.set()

    def wait_termination(self, end=True):
        """
        Waits for all tasks in the queue to be completed.

        Args:
            end (bool): If True, fully stops all worker threads. If False,
                        the pool can be re-enabled by clearing the shutdown signal.
        """
        self.task_queue.join()
        if end is True:
            self.stop_signal.set() 
            for i in xrange(0, self.num_threads):
                self.task_queue.put(None) # Put sentinel values for workers.
                self.num_tasks.release()

            for i in xrange(0, self.num_threads):
                self.threads[i].join()
        else:
            self.shutdown_signal.clear()


class Worker(Thread):
    """A persistent worker thread for the ThreadPool."""

    def __init__(self, task_queue, num_tasks, stop_signal):
        Thread.__init__(self)
        self.task_queue = task_queue
        self.num_tasks = num_tasks
        self.stop_signal = stop_signal

    def run(self):
        """Continuously fetches and runs tasks from the queue."""
        while True:
            self.num_tasks.acquire()
            if self.stop_signal.is_set():
                break

            task = self.task_queue.get()
            task.run()
            self.task_queue.task_done()

class SharedDeviceData(object):
    """
    A container for data and synchronization objects shared across all devices.
    """

    def __init__(self, num_devices):
        """
        Initializes the shared data container.

        Args:
            num_devices (int): The number of devices sharing this data.
        """
        self.num_devices = num_devices
        # Global barrier to sync all master device threads at the end of a timepoint.
        self.timepoint_barrier = CyclicBarrier(num_devices)

        # A dictionary to hold a lock for each unique data location.
        self.location_locks = {}

        # A lock to protect access to the `location_locks` dictionary itself.
        self.ll_lock = Lock()
