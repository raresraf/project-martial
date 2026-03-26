"""
This module implements a Device class representing a simulated entity
that processes scripts and interacts with other devices. It utilizes
threading for concurrent operations and a ThreadPool for managing script
execution. A ReusableBarrierCond (from an external 'barrier' module)
is used for synchronizing devices across timepoints.
"""

from threading import Event, Thread, Lock

from barrier import ReusableBarrierCond
# Changed 'from threadpool import ThreadPool' to 'from queue import Queue' for Python 3 compatibility,
# as ThreadPool is defined within this same file and the original import was likely a remnant or misdirection.
from queue import Queue


class Device(object):
    """
    Represents a simulated device within a larger system, responsible for
    managing sensor data, processing assigned scripts, and coordinating
    with a supervisor and other devices. Each device operates within its
    own thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary where keys are data locations (e.g., sensor IDs)
                                and values are their corresponding data.
            supervisor (Supervisor): An object responsible for overseeing and coordinating
                                     this device and potentially others.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when new scripts have been assigned to this device.
        self.script_received = Event()
        # List to store tuples of (script, location) awaiting processing.
        self.scripts = []
        # Event to signal that the device has completed all script processing for a given timepoint.
        self.timepoint_done = Event()

        # Dictionary to store locks, one for each data location, to ensure
        # thread-safe access to sensor data.
        self.locations_locks = []
        for location in sensor_data:
            self.locations_locks.append((location, Lock()))
        self.locations_locks = dict(self.locations_locks)

        # ReusableBarrierCond instance for device synchronization. Initialized in setup_devices.
        self.barrier = None

        # A dedicated thread for this device to manage its operations.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the synchronization barrier for a group of devices.
        Only device_id 0 initializes the barrier, then shares it with others.

        Args:
            devices (list): A list of all Device objects in the group that will use this barrier.
        """
        # Changed dictionary.has_key to 'in' operator for Python 3 compatibility
        if 0 not in dictionary: # Assuming dictionary is a global variable from the original code that stores barriers
            dictionary[0] = ReusableBarrierCond(len(devices)) # ReusableBarrierCond is from 'barrier' module
            for device in devices:
                if device.device_id != 0:
                    device.barrier = dictionary[0] # Assign the created barrier to other devices

    def assign_script(self, script, location):
        """
        Assigns a script to the device for future execution.
        If 'script' is None, it signals the end of script assignments for a timepoint.

        Args:
            script (object): The script object to be executed, or None to signal timepoint completion.
            location (str): The data location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Indicate that scripts are pending.
        else:
            self.timepoint_done.set() # Indicate that all scripts for this timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves data for a specific location from the device's sensor_data,
        acquiring a lock for thread safety.

        Args:
            location (str): The identifier for the data location.

        Returns:
            any: The data associated with the location, or None if the location is not found.
                 The lock for the location is acquired before returning the data.
        """
        if location in self.sensor_data:
            self.locations_locks[location].acquire() # Acquire lock before accessing data.
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Sets or updates data for a specific location in the device's sensor_data,
        then releases the lock associated with that location.

        Args:
            location (str): The identifier for the data location.
            data (any): The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations_locks[location].release() # Release lock after updating data.

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device's operational thread,
        waiting for it to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the lifecycle and operations of a single Device instance in a separate thread.
    It orchestrates script processing using a thread pool and handles synchronization.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread for a given Device.

        Args:
            device (Device): The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Initialize a ThreadPool to handle the execution of scripts.
        # MAX_THREADS (8) is hardcoded here, which might be a typo or intended.
        self.thread_pool = ThreadPool(8, device)

    def run(self):
        """
        The main loop for the DeviceThread. It continuously checks for neighbors,
        processes scripts assigned to its device via a thread pool, and
        synchronizes with other DeviceThreads using a barrier.
        """
        while True:
            # Retrieve neighbors from the supervisor for interaction.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit loop if supervisor indicates no neighbors (e.g., shutdown signal).

            # Loop to handle script assignment and timepoint completion.
            while True:
                # Wait for a timepoint to be marked as done (all scripts assigned)
                # and ensure no new scripts are actively being assigned.
                if self.device.timepoint_done.wait() and not self.device.script_received.is_set():
                    self.device.timepoint_done.clear() # Reset for next timepoint.
                    self.device.script_received.set() # This line seems redundant or indicates a logic flow issue
                                                    # as script_received should be cleared after processing.
                    break

                # If new scripts have been received, process them.
                if self.device.script_received.is_set():
                    self.device.script_received.clear() # Acknowledge receipt of scripts.

                    # Submit each assigned script to the thread pool for execution.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task(script, location, neighbours)


            # Wait for all tasks in the thread pool to complete for the current timepoint.
            self.thread_pool.tasks_queue.join()

            # Synchronize with other devices using the barrier before proceeding.
            self.device.barrier.wait()

        # After the main loop breaks, gracefully shut down the thread pool.
        self.thread_pool.join_threads()


# Note: The ThreadPool class is defined here, presumably within the same file.
# However, the import statement `from threadpool import ThreadPool` at the top
# indicates it might also be expected to be in a separate file.
# For this commenting task, it's treated as part of this file.


class ThreadPool(object):
    """
    Manages a pool of worker threads to execute tasks (scripts) concurrently.
    Tasks are submitted to a queue and processed by available threads.
    """

    def __init__(self, number_threads, device):
        """
        Initializes the ThreadPool with a specified number of worker threads.

        Args:
            number_threads (int): The number of worker threads to maintain in the pool.
            device (Device): The Device instance associated with this thread pool.
        """
        self.number_threads = number_threads
        # List to hold references to the worker threads.
        self.device_threads = []
        self.device = device
        # A queue to hold tasks (scripts) to be processed by the worker threads.
        self.tasks_queue = Queue(number_threads)

        # Create and start worker threads.
        for _ in range(0, number_threads): # Changed xrange to range for Python 3 compatibility.
            thread = Thread(target=self.apply_scripts)
            self.device_threads.append(thread)

        for thread in self.device_threads:
            thread.start()

    def apply_scripts(self):
        """
        The main function executed by each worker thread in the pool.
        It continuously retrieves tasks from the queue, processes them,
        and marks them as done.
        """
        while True:
            # Retrieve a task (script, location, neighbors) from the queue.
            script, location, neighbours = self.tasks_queue.get()

            # Check for a shutdown signal (None, None, None task).
            if neighbours is None and script is None:
                self.tasks_queue.task_done() # Mark this shutdown task as done.
                return # Terminate the worker thread.

            script_data = []
            # Gather data from neighboring devices (excluding itself).
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            # Gather data from the current device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # If any relevant data was collected, execute the script.
            if script_data != []:
                # Execute the script with the gathered data.
                result = script.run(script_data)

                # Propagate the result to neighboring devices (excluding itself).
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                # Update the data on the current device.
                self.device.set_data(location, result)

            self.tasks_queue.task_done() # Mark the current task as done.

    def submit_task(self, script, location, neighbours):
        """
        Submits a new task to the thread pool's queue.

        Args:
            script (object): The script to be executed.
            location (str): The data location associated with the script.
            neighbours (list): A list of neighboring Device objects.
        """
        self.tasks_queue.put((script, location, neighbours))

    def join_threads(self):
        """
        Initiates a graceful shutdown of all worker threads in the pool.
        It waits for all pending tasks to complete, then sends shutdown signals
        to each thread and waits for them to terminate.
        """
        self.tasks_queue.join() # Wait for all tasks in the queue to be processed.

        # Send a shutdown signal to each worker thread.
        for _ in range(0, len(self.device_threads)): # Changed xrange to range for Python 3 compatibility.
            self.submit_task(None, None, None)

        # Wait for all worker threads to terminate.
        for thread in self.device_threads:
            thread.join()
