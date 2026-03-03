"""
This module defines the core components for a distributed device simulation.

It implements a system where each `Device` acts as a node, managing sensor data,
executing scripts, and coordinating with a supervisor and other devices.
The architecture leverages a `ThreadPool` for concurrent script execution,
and employs a combination of `Event`s, `ReusableBarrierCond`, `Lock`s, and `Semaphore`s
to ensure thread safety and synchronized operations across the distributed system.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond # Assuming ReusableBarrierCond provides a conditional reusable barrier
from thread_pool import ThreadPool     # Assuming thread_pool.py defines the ThreadPool and its WorkerThread


class Device(object):
    """
    Represents a single device (node) in a distributed simulation or data processing system.

    Each device manages its own sensor data, processes assigned scripts, and
    coordinates with a central supervisor and other devices for synchronized operations.
    It uses a dedicated `DeviceThread` for its control logic and delegates script
    execution to a shared `ThreadPool` via `ThreadPool`'s `WorkerThread`s.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique integer identifier for this device.
            sensor_data (dict): A dictionary mapping locations (str) to sensor data (object).
            supervisor (object): An object providing central coordination, typically used
                                 to retrieve information about other devices or global state.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Events for internal synchronization and signaling.
        self.script_received = Event()  # Signals when a new script has been assigned to the device.
        self.scripts = []               # List to store (script, location) tuples assigned to this device.
        self.timepoint_done = Event()   # Signals that all scripts for a specific timepoint have been assigned.

        # The DeviceThread manages the core logic of this device.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Shared resources, typically initialized by a coordinating device (e.g., device_id 0).
        self.barrier = None                 # ReusableBarrierCond for global synchronization across all devices.
        self.devices_synchronized = Event() # Signals when the initial setup of shared resources is complete.
        self.location_semaphores = {}       # Dictionary mapping locations to Semaphores for controlling access to data.
        self.scripts_lock = Lock()          # Lock to protect concurrent access to the `self.scripts` list.
        self.new_location_lock = None       # Lock to protect concurrent creation of new location semaphores.

    def __str__(self):
        """
        Returns a human-readable string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of shared synchronization primitives across all devices.

        Typically, Device with `device_id == 0` acts as a coordinator. It initializes
        the `ReusableBarrierCond`, `location_semaphores` dictionary, and `new_location_lock`,
        then distributes references to these shared objects to all other devices.
        Other devices wait for this setup to complete via `devices_synchronized` event.

        Args:
            devices (list): A list of all Device instances in the distributed system.
        """
        # Block Logic: Device 0 acts as a coordinator for setting up shared resources.
        if self.device_id == 0:
            # Initialize global synchronization primitives.
            barrier = ReusableBarrierCond(len(devices)) # Barrier for all devices.
            location_semaphores = {}                   # Semaphores for controlling location access.
            new_location_lock = Lock()                 # Lock for protecting semaphore creation.

            # Distribute the initialized shared resources to all devices.
            for device in devices:
                device.barrier = barrier
                device.location_semaphores = location_semaphores
                device.new_location_lock = new_location_lock
                device.devices_synchronized.set() # Signal that setup is complete for this device.

        # All devices (including the coordinator) wait for the setup to be signaled.
        self.devices_synchronized.wait()


    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device or signals a timepoint completion.

        If a script object is provided, it's added to the device's internal `scripts` list.
        It also ensures a `Semaphore` exists for the given `location`, creating one if necessary.
        If `script` is `None`, it signals that no more scripts are expected for the current
        timepoint, setting the `timepoint_done` event.

        Args:
            script (object): The script object to be executed, or `None` to signal timepoint completion.
            location (str): The identifier for the data location the script operates on.
        """
        # Critical Section: Protects the creation of new location semaphores.
        self.new_location_lock.acquire()
        if location not in self.location_semaphores:
            # If a semaphore for this location does not exist, create a new one.
            # Default semaphore value is 1, acting as a mutex for that location.
            self.location_semaphores[location] = Semaphore()
        self.new_location_lock.release()

        # Critical Section: Protects concurrent access to the `self.scripts` list.
        self.scripts_lock.acquire()
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location.
        else:
            self.timepoint_done.set() # Signal that all scripts for the current timepoint have been assigned.
        self.script_received.set() # Signal that a script (or timepoint completion) has been received.
        self.scripts_lock.release()


    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from this device's local storage.

        Args:
            location (str): The identifier of the data location.

        Returns:
            object: The sensor data at the given location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location on this device's local storage.

        Args:
            location (str): The identifier of the data location.
            data (object): The new data value to set for the location.
        """
        # Data is only updated if the location already exists in sensor_data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates a graceful shutdown of the device's main `DeviceThread`.
        """
        self.thread.join() # Wait for the DeviceThread to complete its execution.


class DeviceThread(Thread):
    """
    The main thread for a `Device`, responsible for orchestrating timepoint processing.

    It manages synchronization using a global `ReusableBarrierCond`, fetches neighbor
    information from the supervisor, and delegates script execution to a `ThreadPool`.
    It ensures that all scripts for a given timepoint are processed before advancing.
    """

    def __init__(self, device):
        """
        Initializes the `DeviceThread`.

        Args:
            device (Device): The `Device` instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.executor = ThreadPool(8) # Initializes a ThreadPool with 8 worker threads for script execution.

    def run(self):
        """
        The main execution loop for the `DeviceThread`.

        This loop continuously:
        1. Starts the `ThreadPool`'s worker threads.
        2. Retrieves neighbor information from the supervisor. If `None` is returned,
           it signifies a system-wide shutdown, prompting the `ThreadPool` to shut down
           and the `DeviceThread` to terminate.
        3. Processes incoming scripts for the current timepoint. It waits for scripts
           to be assigned, submits them to the `ThreadPool`, and tracks already submitted scripts.
        4. Waits for all scripts submitted to the `ThreadPool` to complete.
        5. Synchronizes with all other devices using the global `ReusableBarrierCond` before
           starting the next timepoint processing cycle.
        """
        self.executor.start_workers() # Start the worker threads in the ThreadPool.
        while True:
            # Block Logic: Retrieve current neighbours and handle system shutdown.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Check for a shutdown signal from the supervisor.
                self.executor.shutdown() # Signal the ThreadPool to shut down its workers.
                break # Exit the timepoint processing loop.

            script_done = {} # Dictionary to keep track of scripts already submitted to the ThreadPool.

            # Critical Section: Protects access to `self.device.scripts` and related events.
            self.device.scripts_lock.acquire()

            # Block Logic: Process incoming scripts for the current timepoint.
            # Continues until `timepoint_done` is set AND no new scripts are pending (`script_received` is clear).
            while not self.device.timepoint_done.isSet() or self.device.script_received.isSet():
                self.device.scripts_lock.release() # Temporarily release lock to allow `assign_script` to run.
                self.device.script_received.wait() # Wait for a new script or timepoint completion signal.
                self.device.script_received.clear() # Clear the event for the next signal.

                # Critical Section: Re-acquire lock to safely access `self.device.scripts`.
                self.device.scripts_lock.acquire()

                # Iterate through all assigned scripts and submit new ones to the thread pool.
                for (script, location) in self.device.scripts:
                    # Check if this (script, location) tuple has already been submitted to the executor.
                    if (script, location) in script_done:
                        continue # Skip if already processed.

                    # Submit the script and relevant context to the ThreadPool for execution.
                    self.executor.submit((self.device, neighbours, script, location))
                    script_done[(script, location)] = True # Mark as submitted.

                # Release lock before potentially waiting again or exiting the inner loop.
                self.device.scripts_lock.release()
                self.device.scripts_lock.acquire() # Re-acquire lock to check loop condition.

            self.device.scripts_lock.release() # Final release of the scripts lock for this timepoint.

            # Block Logic: Wait for all scripts submitted to the ThreadPool to complete.
            self.executor.wait_all()

            # Global Synchronization Point: Wait for all devices to reach this barrier
            # before advancing to the next timepoint.
            self.device.barrier.wait()

            # Reset the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()


"""
This section is from thread_pool.py, which defines the ThreadPool and WorkerThread
classes used by the DeviceThread for concurrent script execution.
"""
from Queue import Queue
from threading import Thread

class ThreadPool(object):
    """
    A simple thread pool implementation for managing a fixed number of worker threads.

    Tasks are submitted to a queue, and worker threads fetch tasks from this queue
    and execute them concurrently.
    """

    def __init__(self, num_threads):
        """
        Initializes the ThreadPool with a specified number of worker threads.

        Args:
            num_threads (int): The number of worker threads to maintain in the pool.
        """
        self._queue = Queue()       # Queue to hold tasks (arguments for worker execution).
        self._num_threads = num_threads # Number of worker threads.
        self._workers = []          # List to hold WorkerThread instances.
        self._init_workers()        # Initialize and create worker threads.

    def _init_workers(self):
        """
        Internal method to create and configure the `WorkerThread` instances for the pool.
        """
        for _ in xrange(self._num_threads):
            self._workers.append(WorkerThread(self._queue))

    def submit(self, args):
        """
        Submits a task to the thread pool's queue for execution by a worker thread.

        Args:
            args (tuple): Arguments to be passed to the `WorkerThread.run` method.
                          Expected to be `(current_device, neighbours, script, location)`.
        """
        self._queue.put(args)

    def wait_all(self):
        """
        Blocks until all tasks currently in the queue have been processed by the worker threads.
        """
        self._queue.join()

    def start_workers(self):
        """
        Starts all worker threads in the pool.
        """
        for worker in self._workers:
            worker.start()

    def shutdown(self):
        """
        Initiates a graceful shutdown of all worker threads in the pool.

        It does this by placing a `None` sentinel value into the queue for each worker,
        which signals the workers to terminate their execution loop.
        """
        for _ in xrange(self._num_threads):
            self._queue.put(None) # Send a termination signal for each worker.
        for worker in self._workers:
            worker.join() # Wait for each worker thread to terminate.


class WorkerThread(Thread):
    """
    A worker thread that continuously fetches tasks from a queue and executes them.

    This `WorkerThread` is designed to be used within a `ThreadPool`. It processes
    scripts at specific data locations, ensuring thread-safe access to data using semaphores.
    """

    def __init__(self, queue):
        """
        Initializes the `WorkerThread`.

        Args:
            queue (Queue): The queue from which the worker fetches tasks.
                           Each task is expected to be a tuple:
                           `(current_device, neighbours, script, location)`.
        """
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        """
        The main execution loop for the `WorkerThread`.

        This loop continuously:
        1. Fetches a task from the queue.
        2. If the task is `None` (a sentinel value), the worker terminates.
        3. Acquires a location-specific semaphore to ensure exclusive access to the data.
        4. Gathers relevant data from the `current_device` and its `neighbours` for the `location`.
        5. Executes the `script` with the collected data.
        6. Updates the data on the `current_device` and its `neighbours` with the script's result.
        7. Releases the location-specific semaphore.
        8. Signals the queue that the task is done.
        """
        while True:
            try:
                # Block Logic: Fetch task from queue.
                # Retrieves a task, which is a tuple containing context for script execution.
                current_device, neighbours, script, location = self.queue.get()
            except TypeError:
                # This block is executed if `None` is received (shutdown signal).
                self.queue.task_done() # Mark the sentinel task as done.
                break # Exit the loop and terminate the worker thread.
            else:
                # Critical Section: Acquire location-specific semaphore to ensure exclusive data access.
                current_device.location_semaphores[location].acquire()

                script_data = [] # List to accumulate data for script execution.

                # Block Logic: Gather data from neighboring devices.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Block Logic: Gather data from the current device.
                data = current_device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Execute script if data is available and update results.
                if script_data != []: # Only run the script if there is data to process.
                    result = script.run(script_data) # Execute the script.

                    # Update data on neighboring devices with the script's result.
                    for device in neighbours:
                        device.set_data(location, result)

                    # Update data on the current device with the script's result.
                    current_device.set_data(location, result)

                # Critical Section Exit: Release location-specific semaphore.
                current_device.location_semaphores[location].release()

                self.queue.task_done() # Signal that the current task is complete.



