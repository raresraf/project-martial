"""
@file device.py
@brief Implements components for a distributed system, likely a simulation or sensor network,
focusing on concurrent data processing, synchronization, and task management.
This module defines Device objects that manage sensor data and execute scripts
using a ThreadPool of worker threads, employing various synchronization
primitives like events, locks, and a reusable barrier.
"""

from threading import Event, Thread, Lock
from operator import attrgetter # Used for efficient attribute access in min() function.
import barrier # Assumed to be an external module defining ReusableBarrierCond, but is part of this file's context.
from pool import ThreadPool # Assumed to be an external module defining ThreadPool, but is part of this file's context.

class ReusableBarrierCond(object):
    """
    @brief Implements a reusable barrier for thread synchronization using a threading.Condition object.
    This barrier allows a fixed number of threads to wait for each other before
    proceeding, and can be reused across multiple synchronization points.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.

        @param num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads  # Counter for threads yet to reach the barrier.
        self.cond = Condition()  # Condition variable for signaling and waiting.


    def wait(self):
        """
        @brief Blocks the calling thread until all 'num_threads' have reached the barrier.
        When the last thread arrives, all waiting threads are notified and the barrier resets.
        """
        self.cond.acquire()  # Acquires the lock associated with the condition variable.
        self.count_threads -= 1  # Decrements the count of threads yet to reach.
        # Conditional Logic: If this is the last thread to reach the barrier.
        if self.count_threads == 0:
            self.cond.notify_all()  # Notifies all waiting threads to proceed.
            self.count_threads = self.num_threads  # Resets the counter for future use.
        else:
            self.cond.wait()  # Waits (releases lock and blocks) until notified.
        self.cond.release()  # Releases the lock.


class Device(object):
    """
    @brief Represents a single device in the distributed system.
    Each device has a unique ID, manages its sensor data, and interacts with a supervisor.
    It uses a ThreadPool to execute assigned scripts concurrently.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary holding sensor readings for different locations.
        @param supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()  # Event to signal when scripts have been assigned for a timepoint.
        self.scripts = []  # List to hold (script, location) tuples assigned to this device.

        self.thread = DeviceThread(self)  # The main thread responsible for this device's lifecycle.
        self.other_devices = []  # List of all other devices in the system.

        self.gdevice = None  # Reference to the "global" device (device with the minimum ID).
        self.gid = None      # ID of the "global" device.

        self.glocks = {}     # Dictionary of global locks, typically one per data location.

        self.barrier = None  # Reference to the global ReusableBarrierCond.

        self.threadpool = None # Reference to the ThreadPool for executing scripts.

        self.nthreads = 8    # Number of worker threads in the ThreadPool.

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return str: A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs initial setup for the device, including identifying the global device,
        initializing global locks and barriers (if this is the global device), and
        creating its ThreadPool.

        @param devices (list): A list of all Device instances in the system.
        """
        self.other_devices = devices  # Stores reference to all devices.

        # Identifies the "global" device (device with the minimum ID) using `attrgetter` for efficiency.
        self.gdevice = min(devices, key=attrgetter('device_id'))
        self.gid = self.gdevice.device_id # Stores the ID of the global device.

        # Conditional Logic: Only the "global" device (the one with the minimum ID) performs global setup.
        if self.device_id == self.gid:
            # Block Logic: Collects all unique data locations across all devices.
            list_loc = []
            for dev in self.other_devices:
                for key, _ in dev.sensor_data.iteritems(): # Python 2 .iteritems() - in Python 3, this would be .items()
                    list_loc.append(key)
            list_nodup = list(set(list_loc)) # Removes duplicate locations.

            # Block Logic: Creates a Lock for each unique data location globally.
            locks = {}
            for loc in list_nodup:
                locks[loc] = Lock()
            self.glocks = locks # Assigns the globally shared locks.
            
            # Initializes the global barrier for all devices.
            self.barrier = ReusableBarrierCond(len(self.other_devices))

        # Initializes a ThreadPool for this device.
        self.threadpool = ThreadPool(self.nthreads)

        # Starts the main DeviceThread.
        self.thread.start()


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location or signals timepoint completion.

        @param script (callable): The script (function or object with a run method) to execute.
                                  If None, it signals that script assignment for the timepoint is done.
        @param location (int): The identifier of the data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location)) # Appends the script and its location.
        else:
            self.script_received.set() # Signals that script assignment for the timepoint is complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location (int): The identifier of the data location.
        @return any: The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location (int): The identifier of the data location to update.
        @param data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's main operational thread.
        Note: The ThreadPool workers are shut down separately via `device.threadpool.end()`.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main thread of execution for a Device.
    It is responsible for fetching neighbor information, coordinating the processing
    of scripts using a ThreadPool, and managing synchronization across devices.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device (Device): The Device object this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop of the DeviceThread.
        It continuously fetches neighbor information, waits for scripts to be assigned,
        submits them to its thread pool, waits for thread pool completion, clears events,
        and then synchronizes with other devices via a global barrier.
        """
        while True:
            # Block Logic: Fetches neighbor devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Conditional Logic: If no neighbors are returned (supervisor signals shutdown), terminates.
            if neighbours is None:
                self.device.threadpool.end() # Signals the thread pool to terminate its workers.
                break # Terminates the thread.

            # Synchronization: Waits until scripts for the current timepoint have been assigned.
            self.device.script_received.wait()

            # Block Logic: Adds all assigned scripts to the thread pool as tasks.
            for (script, location) in self.device.scripts:
                # Task includes (device, script, location, neighbours) for worker processing.
                self.device.threadpool.add_task((self.device, script, location, neighbours))
            self.device.threadpool.finish_tasks() # Waits for all tasks in the thread pool to complete.

            # Clears the script_received event for the next timepoint.
            self.device.script_received.clear()

            # Synchronization: Waits at the global barrier (managed by the gdevice).
            # This synchronizes all devices at the end of a timepoint.
            self.device.gdevice.barrier.wait()


from threading import Lock, Thread
from Queue import Queue # In Python 3, this is `from queue import Queue`.

class WorkerThread(Thread):
    """
    @brief A worker thread that continuously processes tasks from a ThreadPool's queue.
    Each task involves running a script, collecting data from neighbors, and updating data.
    """

    def __init__(self, threadpool):
        """
        @brief Initializes a new WorkerThread instance.

        @param threadpool (ThreadPool): The ThreadPool this worker belongs to.
        """
        Thread.__init__(self, name="worker")
        self.threadpool = threadpool
        self.start() # Starts the worker thread immediately upon creation.

    def run(self):
        """
        @brief The main execution loop for a WorkerThread.
        It continuously retrieves tasks from the ThreadPool's queue,
        processes them, and performs necessary data updates and synchronization.
        """
        while True:
            # Conditional Logic: Checks for a termination signal from the ThreadPool.
            if self.threadpool.stop:
                break # Terminates the worker thread.

            current_task = None

            # Acquires task_lock to safely get a task from the queue.
            with self.threadpool.task_lock:
                # Conditional Logic: Attempts to get a task without blocking.
                if self.threadpool.tasks.qsize() > 0:
                    current_task = self.threadpool.tasks.get_nowait() # Non-blocking task retrieval.

            # Conditional Logic: If a task was successfully retrieved.
            if current_task is not None:
                (device, script, location, neighbours) = current_task

                # Synchronization: Acquires the global lock for the specific data location.
                # This ensures only one thread can modify/access data at this location at a time.
                with device.gdevice.glocks[location]:
                    script_data = [] # List to collect data for the script.
                    
                    # Block Logic: Collects data from neighboring devices.
                    for dev in neighbours:
                        data = dev.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    # Collects data from its own device.
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    # Conditional Logic: If any data was collected, executes the script and propagates results.
                    if script_data != []:
                        result = script.run(script_data) # Executes the script.
                        
                        # Block Logic: Propagates the new data to all neighboring devices.
                        for dev in neighbours:
                            dev.set_data(location, result)
                        
                        device.set_data(location, result) # Updates data on its own device.

                # Signals that the current task is complete in the queue.
                self.threadpool.tasks.task_done()


class ThreadPool(object):
    """
    @brief Manages a pool of worker threads to execute tasks concurrently.
    Tasks (scripts) are submitted to a queue, and worker threads pick them up for processing.
    """

    def __init__(self, size):
        """
        @brief Initializes the ThreadPool.

        @param size (int): The number of worker threads to create in the pool.
        """
        self.size = size
        
        self.tasks = Queue() # A queue to hold tasks for worker threads.
        
        self.workers = [] # List to hold the worker Thread objects.
        
        self.task_lock = Lock() # Lock to protect access to the tasks queue (used for `get_nowait`).
        
        self.stop = False # Flag to signal worker threads to stop.

        # Block Logic: Creates and starts the specified number of worker threads.
        for _ in xrange(self.size): # Python 2 `xrange` - in Python 3, this would be `range`.
            self.workers.append(WorkerThread(self))

    def add_task(self, task):
        """
        @brief Submits a new task (script with its context) to the thread pool queue.

        @param task (tuple): A tuple containing (device, script, location, neighbours).
        """
        with self.task_lock: # Protects queue operations with a lock.
            self.tasks.put(task)

    def finish_tasks(self):
        """
        @brief Blocks until all tasks currently in the queue have been processed.
        """
        self.tasks.join()

    def end(self):
        """
        @brief Signals all worker threads to terminate and waits for their completion.
        """
        self.tasks.join() # Ensures all current tasks are finished.
        self.stop = True  # Sets the stop flag.
        # Block Logic: Submits a termination signal to each worker thread.
        # This is necessary for `get()` to unblock and see the `stop` flag.
        for thread in self.workers:
            # Submitting None task to unblock workers waiting on `tasks.get()`
            self.tasks.put((None, None, None, None))
        
        for thread in self.workers: # Waits for all worker threads to join (terminate).
            thread.join()
