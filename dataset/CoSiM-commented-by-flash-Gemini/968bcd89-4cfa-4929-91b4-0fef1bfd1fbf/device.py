"""
This module simulates a distributed device system where each Device manages
its own execution thread and utilizes a thread pool for concurrent script
processing. It incorporates a barrier synchronization mechanism to coordinate
activities across multiple devices, facilitating parallel data processing
and communication within a simulated environment.
"""

from threading import Event, Thread, Lock
from barrier import Barrier # Assuming 'barrier.py' defines a Barrier class for synchronization.
from thread_pool import ThreadPool # Assuming 'thread_pool.py' defines a ThreadPool class.
from Queue import Queue # Python 2.x Queue, for Python 3.x this would be 'queue.Queue'.


class Device(object):
    """
    Represents a single device within the simulated distributed system.
    Each device has a unique ID, sensor data, interacts with a supervisor,
    and manages script execution and synchronization with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing sensor readings at various locations.
            supervisor (object): A reference to the central supervisor managing device interactions.
        """

        self.device_id = device_id
        self.num_threads = 8 # Number of worker threads this device's thread pool will use.
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to store assigned scripts and their associated locations.
        self.timepoint_done = Event() # Event to signal when processing for a timepoint is complete.
        self.barrier = None # Barrier instance for global synchronization, initialized by the master device.
        self.location_lock = {} # Dictionary to store locks for protecting sensor data at specific locations.
        self.thread = DeviceThread(self) # Dedicated thread for the device's main operational loop.
        self.thread.start() # Starts the device's main execution thread.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the synchronization mechanisms (barrier and location-specific locks)
        across all devices in the system. This method is typically called once by a master device.

        Args:
            devices (list): A list of all Device instances participating in the simulation.
        """

        # Block Logic: Ensures that the barrier and shared locks are initialized only once by the master device (device_id == 0).
        # Pre-condition: This method is expected to be called on the master device.
        if self.device_id == 0:
            # Block Logic: Initializes the global barrier if it hasn't been set yet.
            if self.barrier is None:
                self.barrier = Barrier(len(devices)) # Functional Utility: Creates a new barrier for all participating devices.

            # Block Logic: Initializes location-specific locks by iterating through all devices and their sensor data.
            # Invariant: 'location' keys in 'self.location_lock' correspond to unique data locations.
            for device in devices:
                for location in device.sensor_data:
                    if location not in self.location_lock:
                        self.location_lock[location] = Lock() # Functional Utility: Creates a new lock for each unique location.
                # Functional Utility: Propagates the initialized barrier and location locks to all devices.
                device.barrier = self.barrier
                device.location_lock = self.location_lock

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.
        If script is None, it signals the completion of script assignments for the current timepoint.

        Args:
            script (object or None): The script object to execute, or None to signal completion.
            location (str): The data location relevant to the script.
        """

        # Block Logic: Differentiates between actual script assignment and a signal for timepoint completion.
        if script is not None:
            self.scripts.append((script, location)) # Functional Utility: Adds the script and location to the device's pending scripts.
        else:
            self.timepoint_done.set() # Functional Utility: Signals that all scripts for the current timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (str): The specific location for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (str): The specific location to update.
            data (any): The new data value for the location.
        """

        # Block Logic: Ensures data is updated only if the location exists in the device's sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device, waiting for its main execution thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device. It orchestrates the process of
    fetching neighbor information, submitting scripts to a thread pool for execution,
    and synchronizing with other devices using a barrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance that this thread controls.
        """

        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device # Reference to the associated Device instance.
        self.thread_pool = ThreadPool(device, self.device.num_threads) # Manages a pool of worker threads for script execution.

    def run(self):
        """
        The main loop for the device thread. It manages the lifecycle of script
        processing for each timepoint, from receiving scripts to barrier synchronization
        and eventual shutdown.
        """

        self.thread_pool.start_threads() # Functional Utility: Starts the worker threads in the thread pool.

        # Block Logic: Main operational loop for the device, processing timepoints.
        # Invariant: Continues indefinitely until the supervisor signals termination.
        while True:
            # Functional Utility: Retrieves information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Exits the loop if no neighbors are returned (signaling system shutdown or completion).
            # Pre-condition: 'neighbours' being None indicates a termination signal.
            if neighbours is None:
                break

            # Functional Utility: Waits until scripts have been assigned for the current timepoint.
            self.device.timepoint_done.wait()

            # Block Logic: Queues each assigned script for execution by the thread pool.
            # Invariant: Each item in 'self.device.scripts' is a (script, location) tuple.
            for script, location in self.device.scripts:
                self.thread_pool.queue.put((script, location, neighbours)) # Functional Utility: Adds a script task to the thread pool queue.

            
            # Functional Utility: Blocks until all tasks in the thread pool queue are processed.
            self.thread_pool.queue.join()
            # Functional Utility: Synchronizes with other devices, waiting until all have completed their timepoint processing.
            self.device.barrier.wait()
            # Functional Utility: Resets the 'timepoint_done' event for the next timepoint.
            self.device.timepoint_done.clear()

        # Block Logic: Signals the thread pool to shut down its worker threads.
        # Invariant: 'self.device.num_threads' determines how many shutdown signals are sent.
        for _ in range(self.device.num_threads):
            self.thread_pool.queue.put((None, None, None)) # Functional Utility: Enqueues shutdown signals for worker threads.

        self.thread_pool.end_threads() # Functional Utility: Waits for all worker threads to terminate.


class ThreadPool(object):
    """
    Manages a pool of worker threads responsible for executing scripts
    assigned to a device. It uses a queue to distribute tasks to available threads.
    """

    def __init__(self, device, num_threads):
        """
        Initializes the ThreadPool.

        Args:
            device (Device): The Device instance that owns this thread pool.
            num_threads (int): The number of worker threads in the pool.
        """

        self.queue = Queue(num_threads) # A bounded queue to hold script execution tasks.
        self.device = device # Reference to the owning Device instance.
        self.threads = [] # List to store references to the worker threads.
        self.num_threads = num_threads # The total number of threads in the pool.

    def start_threads(self):
        """
        Creates and starts the specified number of worker threads, each
        configured to execute tasks from the internal queue.
        """

        # Block Logic: Creates and appends worker threads to the internal list.
        # Invariant: 'self.num_threads' dictates the size of the thread pool.
        for _ in range(self.num_threads):
            self.threads.append(Thread(target=self.run)) # Functional Utility: Creates a new worker thread.



        # Block Logic: Starts each worker thread.
        for thread in self.threads:
            thread.start()

    def run(self):
        """
        The main loop for each worker thread in the pool. It continuously
        fetches tasks from the queue, executes the assigned script, and
        signals task completion.
        """

        # Block Logic: Continuously processes tasks from the queue until a shutdown signal is received.
        # Invariant: 'script' and 'location' being None indicates a shutdown request.
        while True:
            script, location, neighbours = self.queue.get() # Functional Utility: Retrieves a task from the queue.

            # Block Logic: Checks for the shutdown signal (None, None, None).
            if script is None and location is None:
                return # Functional Utility: Terminates the worker thread.

            self.run_script(script, location, neighbours) # Functional Utility: Executes the assigned script.
            self.queue.task_done() # Functional Utility: Marks the current task as complete in the queue.

    def run_script(self, script, location, neighbours):
        """
        Executes a single script, gathering necessary data from the current device
        and its neighbors, and then updates the sensor data.

        Args:
            script (object): The script object to execute.
            location (str): The data location relevant to the script.
            neighbours (list): A list of neighboring Device instances.
        """

        # Block Logic: Acquires a lock for the specific location to ensure exclusive access to data during script execution.
        with self.device.location_lock[location]:
            script_data = [] # Data collected for the current script's execution.

            # Block Logic: Gathers data from neighboring devices for the script.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Functional Utility: Gathers data from the current device for the script.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script only if there is data to process.
            if script_data != []:
                result = script.run(script_data) # Functional Utility: Executes the script and obtains the result.

                # Block Logic: Updates the sensor data on all neighboring devices with the script's result.
                for device in neighbours:
                    device.set_data(location, result)

                # Functional Utility: Updates the sensor data on the current device with the script's result.
                self.device.set_data(location, result)

    def end_threads(self):
        """
        Waits for all worker threads in the pool to terminate.
        """

        # Block Logic: Joins each worker thread, blocking until it completes its execution.
        for thread in self.threads:
            thread.join()
