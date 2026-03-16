# -*- coding: utf-8 -*-
"""
This module implements a simulation of a distributed sensor network.

It defines a `Device` class representing a sensor node, which operates concurrently
with other devices. The devices execute scripts on sensor data, communicate with
their neighbors, and synchronize their operations using a thread pool and barriers.
This setup is designed to model a parallel, distributed computation environment.

Classes:
    Device: Represents a single device in the network.
    DeviceThread: The main control thread for a Device.
    WorkerThread: A thread from a WorkPool that executes computation tasks.
    WorkPool: A thread pool for managing and executing concurrent tasks.
"""

from threading import Event, Thread, Lock
from pool import WorkPool
from reusable_barrier import ReusableBarrier

class Device(object):
    """
    Represents a single device in the distributed sensor network.

    Each device manages its own sensor data, executes assigned scripts, and
    communicates with other devices. It uses a thread-based model to perform
    its operations concurrently.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data.
        supervisor (Supervisor): An object that oversees the device's operation.
        script_received (Event): An event to signal the arrival of new scripts.
        scripts (list): A list of scripts to be executed.
        timepoint_done (Event): An event to signal the completion of a timepoint.
        other_devs (list): A list of other devices in the network.
        slock (Lock): A lock for serializing access to sensor data.
        barrier (ReusableBarrier): A barrier for synchronizing with other devices.
        process (Event): An event to control the device's processing loop.
        global_thread_pool (WorkPool): A shared thread pool for executing tasks.
        glocks (dict): A dictionary of shared locks for sensor data locations.
        thread (DeviceThread): The main execution thread for this device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor (Supervisor): The supervisor for this device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Event to signal that a script has been received and is ready for processing.
        self.script_received = Event()
        self.scripts = []

        self.timepoint_done = Event()
        self.other_devs = []
        # Lock for accessing the device's local sensor data.
        self.slock = Lock()

        self.barrier = None
        self.process = Event()

        self.global_thread_pool = None
        # Global locks for each sensor data location, shared across devices.
        self.glocks = {}

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's connections with other devices in the network.

        This method initializes shared resources like global locks, the thread pool,
        and the synchronization barrier. The device with the lowest ID is designated
        as the leader to initialize these shared resources.

        Args:
            devices (list): A list of all devices in the network.
        """
        self.other_devs = devices
        # The first device in the list acts as the coordinator for setup.
        if self.device_id == self.other_devs[0].device_id:
            locks = {}
            for loc in self.sensor_data:
                locks[loc] = Lock()
            dev_cnt = len(devices)
            self.glocks = locks
            self.barrier = ReusableBarrier(dev_cnt)
            self.global_thread_pool = WorkPool(16)
        else:
            # Other devices inherit the shared resources from the coordinator.
            for loc in self.sensor_data:
                self.other_devs[0].glocks[loc] = Lock()
            self.glocks = self.other_devs[0].glocks
            self.global_thread_pool = self.other_devs[0].global_thread_pool
            self.barrier = self.other_devs[0].barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        If the script is None, it signals that all scripts for the current
        timepoint have been assigned.

        Args:
            script (Script): The script to execute.
            location (str): The sensor data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is a sentinel value to indicate all scripts are assigned.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location in a thread-safe manner.

        Args:
            location (str): The location of the sensor data to retrieve.

        Returns:
            The sensor data at the given location, or None if the location
            is not found.
        """
        ret = None
        with self.slock:
            if location in self.sensor_data:
                ret = self.sensor_data[location]
        return ret

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location in a thread-safe manner.

        Args:
            location (str): The location of the sensor data to update.
            data: The new data to be stored.
        """
        with self.slock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main execution thread for a Device.

    This thread manages the device's lifecycle, which consists of waiting for
    scripts, dispatching them to a worker pool for execution, and synchronizing
    with other devices at the end of each timepoint.
    """
    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread.

        It continuously waits for work from the supervisor. When scripts are
        received, they are dispatched to the global thread pool. After all

        scripts for a timepoint are processed, it synchronizes with other
        devices using a barrier.
        """
        while True:
            # Get the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbours indicates a shutdown signal.
                # The lead device is responsible for shutting down the global thread pool.
                if self.device.device_id is self.device.other_devs[0].device_id:
                    self.device.global_thread_pool.end()
                break

            # Wait until all scripts for the current timepoint have been assigned.
            self.device.script_received.wait()

            # Dispatch each script as a work item to the global thread pool.
            for (script, location) in self.device.scripts:
                self.device.global_thread_pool.work((self.device,
                									script,
                									location,
                									neighbours))

            # Wait for all work items in the current batch to complete.
            self.device.global_thread_pool.finish_work()
            self.device.script_received.clear()
            # Synchronize with all other devices before proceeding to the next timepoint.
            self.device.barrier.wait()


from threading import Lock, Event, Semaphore, Thread

class WorkerThread(Thread):
    """
    A worker thread that executes tasks from a shared WorkPool.
    """
    def __init__(self, i, parent_work_pool):
        """
        Initializes a WorkerThread.

        Args:
            i (int): The identifier for this worker thread.
            parent_work_pool (WorkPool): The pool this worker belongs to.
        """
        Thread.__init__(self, name="WorkerThread%d" % i)
        self.pool = parent_work_pool

    def run(self):
        """
        The main loop for the worker thread.

        It continuously waits for tasks from the pool, executes them, and
        signals completion.
        """
        while True:
            # Wait for a task to become available.
            self.pool.task_sign.acquire()
            if self.pool.stop:
                break
            current_task = (None, None, None, None)
            with self.pool.task_lock:
                task_count = len(self.pool.tasks_list)
                
                if task_count > 0:
                    # Dequeue a task.
                    current_task = self.pool.tasks_list[0]
                    self.pool.tasks_list = self.pool.tasks_list[1:]
                
                if task_count == 1:
                    # If this was the last task, signal that the pool is now idle.
                    self.pool.no_tasks.set()

            if current_task is not None:
                (current_device, script, location, neighbourhood) = current_task
                # Acquire a lock for the specific sensor location to ensure data consistency.
                with current_device.glocks[location]:
                    common_data = []
                    
                    # Gather data from all neighboring devices for the given location.
                    for neighbour in neighbourhood:
                        data = neighbour.get_data(location)
                        if data is not None:
                            common_data.append(data)
                    
                    # Also include the current device's data.
                    data = current_device.get_data(location)
                    if data is not None:
                        common_data.append(data)

                    if common_data != []:
                         
                        # Execute the script with the aggregated data.
                        result = script.run(common_data)
                        # Broadcast the result to all neighbors and the current device.
                        for neighbour in neighbourhood:
                            neighbour.set_data(location, result)
                        
                        current_device.set_data(location, result)

class WorkPool(object):
    """
    A simple thread pool for managing a set of worker threads.
    """
    def __init__(self, size):
        """
        Initializes the WorkPool.

        Args:
            size (int): The number of worker threads in the pool.
        """
        self.size = size

        self.tasks_list = [] 
        self.task_lock = Lock()
        self.task_sign = Semaphore(0)
        self.no_tasks = Event()
        self.no_tasks.set()
        self.stop = False

        self.workers = []
        for i in xrange(self.size):
            worker = WorkerThread(i, self)
            self.workers.append(worker)

        for worker in self.workers:
            worker.start()

    def work(self, task):
        """
        Adds a task to the work queue.

        Args:
            task: The task to be executed.
        """
        with self.task_lock:
            self.tasks_list.append(task)
            self.task_sign.release()
            if self.no_tasks.is_set():
                self.no_tasks.clear()

    def finish_work(self):
        """Blocks until all tasks in the queue are completed."""
        self.no_tasks.wait()

    def end(self):
        """Stops all worker threads and joins them."""
        self.finish_work()
        self.stop = True
        for thread in self.workers:
            self.task_sign.release()
        for thread in self.workers:
            thread.join()
