


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `MyBarrier`: A reusable barrier synchronization mechanism using `threading.Condition`.
- `MyWorkerThread`: A worker thread managed by `MyThreadPool` that processes individual tasks.
- `MyThreadPool`: A custom thread pool manager for distributing tasks to `MyWorkerThread`s.
- `Device`: Represents a single device, managing its sensor data and orchestrating operations.
- `DeviceThread`: The main orchestrating thread for a `Device`, leveraging `MyThreadPool`
  to execute scripts concurrently and handling inter-device synchronization.

The system utilizes various `threading` primitives (`Thread`, `Condition`, `Event`, `Lock`)
and `Queue` for intricate synchronization and task management.
"""

from threading import Thread, Condition, Event, Lock
from Queue import Queue

def stop():
    """
    A sentinel function used to signal a `MyWorkerThread` to terminate its execution.
    It performs no operation.
    """
    return

class MyBarrier(object):
    """
    A reusable barrier synchronization mechanism for multiple threads using `threading.Condition`.
    Threads wait at this barrier until a specified number of threads (`num_threads`) have arrived.
    Once all threads arrive, they are all notified and released simultaneously.
    The barrier can then be reused for subsequent synchronization points.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate
                               in the barrier synchronization.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Counter for threads currently waiting at the barrier.
        self.cond = Condition() # The condition variable used for thread synchronization.
    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have arrived. Once all threads are present, they are all released.
        """
        
        self.cond.acquire() # Acquire the condition variable's intrinsic lock.
        self.count_threads -= 1 # Decrement the count of threads waiting.
        if self.count_threads == 0: # If this is the last thread to arrive:
            self.cond.notify_all() # Notify all waiting threads to resume.
            self.count_threads = self.num_threads # Reset the counter for the next use of the barrier.
        else:
            self.cond.wait() # If not the last thread, wait to be notified.
        self.cond.release() # Release the condition variable's intrinsic lock.


class MyWorkerThread(Thread):
    """
    A worker thread designed to run tasks submitted to a `MyThreadPool`.
    It continuously fetches tasks (function and parameters) from a shared queue,
    executes them, and signals completion. It terminates upon receiving a `stop()`
    sentinel function.
    """
    
    def __init__(self, tasks_list, lock):
        """
        Initializes a `MyWorkerThread` instance.

        Args:
            tasks_list (Queue): The shared queue from which this worker retrieves tasks.
            lock (Lock): A lock passed to the worker, likely used for specific signaling
                         during shutdown.
        """
        Thread.__init__(self)
        self.tasks_list = tasks_list # Reference to the shared task queue.
        self.daemon = True # Sets the thread as a daemon, meaning it will exit when the main program exits.
        self.stop = False # Flag (unused in current run logic)
        self.lock = lock # Lock for coordination, particularly during shutdown.
        self.start() # Automatically start the thread upon initialization.

    def run(self):
        while True:
            function, params = self.tasks_list.get() # Inline: Retrieve a task (function and its parameters) from the queue.
            
            # Block Logic: Check for the `stop` sentinel function to terminate the worker.
            if function is stop:
                self.tasks_list.task_done() # Inline: Signal that this task (the stop sentinel) is done.
                self.lock.release() # Release the lock, likely signaling MyThreadPool.wait() that this worker has processed its stop.
                break # Terminate the worker thread.

            # Block Logic: Execute the retrieved function with its parameters.
            function(*params) # Unpack parameters and call the function.

            self.tasks_list.task_done() # Inline: Signal that the current task has been completed.

class MyThreadPool(object):
    """
    A custom thread pool implementation for executing tasks concurrently.
    It manages a fixed number of `MyWorkerThread` instances, distributing tasks
    (functions and their arguments) via a shared queue. It also provides
    mechanisms to add tasks and wait for their completion, including graceful shutdown.
    """
    
    def __init__(self, no_threads):
        """
        Initializes a `MyThreadPool` with a specified number of worker threads.

        Args:
            no_threads (int): The number of `MyWorkerThread` instances to create in the pool.
        """
        self.no_threads = no_threads # Number of worker threads.
        self.tasks_list = Queue(no_threads) # A queue to hold tasks, bounded by the number of threads.
        self.worker_list = [] # List to hold references to the spawned `MyWorkerThread` instances.
        self.lock = Lock() # A lock, primarily used during shutdown to coordinate with workers.
        
        # Inline: Create and start `no_threads` MyWorkerThread instances.
        for _ in xrange(no_threads):
            self.worker_list.append(MyWorkerThread(self.tasks_list, self.lock))

    def add(self, function, *params):
        """
        Adds a new task (a function and its parameters) to the thread pool's queue.
        The task will be picked up by an available worker thread.

        Args:
            function (callable): The function to be executed by a worker.
            *params: Arbitrary positional arguments to be passed to the function.
        """
        self.tasks_list.put((function, params)) # Adds the task as a tuple (function, params) to the queue.

    def wait(self):
        """
        Waits for all tasks currently in the queue to be completed by the worker threads,
        and then initiates a graceful shutdown of all worker threads.
        """
        self.tasks_list.join() # Inline: Blocks until all items in the queue have been gotten and processed (task_done called).
        for i in xrange(self.no_threads):
            self.lock.acquire() # Acquire the lock, signals that a worker will process its stop.
            self.add(stop, None) # Add a `stop` sentinel task to each worker's queue to trigger termination.
        for i in xrange(self.no_threads):
            self.worker_list[i].join() # Wait for each worker thread to fully terminate.



class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution through its `DeviceThread`
    and `MyThreadPool`. It uses per-location locks for data consistency.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for the device.
            supervisor (object): A reference to a supervisor object for inter-device communication.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are assigned.
        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).
        self.timepoint_done = Event() # Event to signal that the current timepoint's processing is complete.
        self.thread = DeviceThread(self) # The main orchestrating thread for this device.
        self.thread.start() # Start the DeviceThread.

        self.barrier = None # Placeholder for the global MyBarrier, set in setup_devices.
        
        self.locks = [] # List of shared locks for specific data locations.
        self.no_locations = self.supervisor.supervisor.testcase.num_locations # Number of distinct data locations.
        
        self.pool = MyThreadPool(8) # Initializes a custom thread pool with 8 worker threads.

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes a global barrier and shared location-specific locks,
        then distributes them to all devices. This method is designed to be
        called only by the device with `device_id == 0`. It creates a single
        `MyBarrier` and a list of `Lock`s for each unique data location
        found across all devices.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        # Block Logic: Only the device with device_id 0 performs this setup.
        if self.device_id == 0:
            # Inline: Creates a global MyBarrier for synchronization among all DeviceThreads.
            barrier = MyBarrier(len(devices))

            # Block Logic: Initializes a global list of locks, one for each data location.
            for i in xrange(self.no_locations):
                self.locks.append(Lock())

            # Block Logic: Distributes the created global barrier and location locks to all devices.
            for i in xrange(len(devices)):
                devices[i].barrier = barrier
                devices[i].locks = self.locks # All devices share the same list of locks.

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals timepoint completion.
        If a script is provided, it's appended to the device's internal script list.
        If `script` is None, it signals that script assignments for the current
        timepoint are complete by setting `timepoint_done`.

        Args:
            script (object): The script object to be executed, or None to signal timepoint completion.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
        else:
            self.timepoint_done.set() # If script is None, signal that script assignments for the timepoint are done.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's `sensor_data` dictionary.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location \
                in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location in this device's `sensor_data` dictionary.
        The data is updated only if the location exists in the `sensor_data`.

        Args:
            location (int): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device.
        It waits for its main `DeviceThread` to complete, which in turn
        will signal its associated `MyThreadPool` to shut down gracefully.
        """
        self.thread.join() # Wait for the DeviceThread to finish its execution.


class DeviceThread(Thread):
    """
    The main orchestrating thread for a `Device`.
    This thread fetches neighbor information, distributes scripts to the device's
    `MyThreadPool` for concurrent execution, and manages timepoint synchronization
    using a global barrier.
    """

    def __init__(self, device):
        """
        Initializes a `DeviceThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_script(self, params):
        """
        Executes a script using collected data from neighbors and the local device.
        This method is designed to be passed as a task to the `MyThreadPool`.

        Args:
            params (tuple): A tuple containing (neighbours, script, location).
                            - neighbours (list): List of neighboring devices.
                            - script (object): The script object to execute.
                            - location (int): The data location identifier.
        """
        neighbours, script, location = params # Unpack parameters.
        self.device.locks[location].acquire() # Inline: Acquire the location-specific lock for data consistency.

        script_data = [] # List to collect input data for the script.
        
        # Block Logic: Collect data from all neighboring devices at the specified location.
        for device in neighbours:
            data = device.get_data(location) # Get data from the neighbor.
            if data is not None:
                script_data.append(data) # Add to script input if available.

        # Block Logic: Collect data from this worker's own parent device at the specified location.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data) # Add to script input if available.

        # Block Logic: If input data is available, execute the script and update device data.
        if script_data != []:
            result = script.run(script_data) # Inline: Execute the script with the collected data.
            
            # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.
            for device in neighbours:
                device.set_data(location, result) # Update neighbor's data.
            
            self.device.set_data(location, result) # Update this device's own data.

        self.device.locks[location].release() # Inline: Release the location-specific lock.

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        It continuously fetches neighbor data, waits for timepoint readiness,
        submits scripts to the `MyThreadPool` for concurrent execution,
        and synchronizes with other devices using a global barrier.
        """
        
        while True:
            # Block Logic: Fetch neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Inline: If `neighbours` is None, it signals termination for the device.
            if neighbours is None:
                self.device.pool.wait() # Signal the thread pool to shut down gracefully and wait for it.
                break # Exit the main loop, terminating the DeviceThread.

            # Block Logic: Wait for the `timepoint_done` event to be set,
            # indicating that all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Submit assigned scripts as tasks to the `MyThreadPool`.
            for (script, location) in self.device.scripts:
                params = (neighbours, script, location) # Bundle parameters for `run_script`.
                self.device.pool.add(self.run_script, params) # Add the task to the thread pool.
            
            self.device.timepoint_done.clear() # Clear the event for the next timepoint.
            self.device.barrier.wait() # Block Logic: Synchronize with other devices at the global barrier.
