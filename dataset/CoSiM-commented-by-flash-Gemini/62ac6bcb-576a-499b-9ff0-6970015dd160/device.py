"""
@62ac6bcb-576a-499b-9ff0-6970015dd160/device.py
@brief Implements a simulated device for distributed sensor data processing using threads and a thread pool.

This module defines the core components for a distributed system where each 'Device'
can process sensor data, interact with neighboring devices, and execute scripts
assigned to it. It utilizes threading primitives for synchronization and data structures
to represent products and carts.
"""

from threading import Event, Thread, Lock

from barrier import ReusableBarrierCond
# from threadpool import ThreadPool # This import is commented out as ThreadPool is defined in this file.


class Device(object):
    """
    @brief Represents a single simulated device in a distributed system.

    Manages sensor data, interacts with a supervisor, and executes scripts
    assigned to it. It uses threading events for coordination and locks for
    managing access to its sensor data.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor data values.
            supervisor (Supervisor): An object responsible for overseeing devices
                                     and providing information like neighbors.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Functional Utility: Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        # Functional Utility: List to store incoming scripts and their associated locations.
        self.scripts = []
        # Functional Utility: Event to signal that all scripts for the current timepoint have been processed.
        self.timepoint_done = Event()

        # Functional Utility: Dictionary to hold locks for each sensor data location,
        # ensuring thread-safe access to sensor data.
        self.locations_locks = []

        # Block Logic: Initializes a lock for each sensor data location to protect against race conditions.
        # Invariant: Each location in sensor_data will have a corresponding lock.
        for location in sensor_data:
            self.locations_locks.append((location, Lock()))

        self.locations_locks = dict(self.locations_locks)

        # Functional Utility: A barrier for coordinating multiple devices, initialized for device 0.
        self.barrier = None

        # Functional Utility: The dedicated thread for this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the reusable barrier for all devices.

        This method is intended to be called by device 0 to initialize the barrier
        and assign it to all other devices.

        Args:
            devices (list): A list of all Device instances in the system.
        """
        # Conditional Logic: Ensures the barrier is initialized only once by the lead device (device_id 0).
        if self.device_id == 0:
            # Functional Utility: Initializes a reusable barrier for coordinating all devices.
            self.barrier = ReusableBarrierCond(len(devices))
            # Block Logic: Assigns the initialized barrier to all other devices.
            # Pre-condition: 'devices' is a list of all Device objects.
            # Invariant: Each device (except device 0) will be assigned the shared barrier.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed on sensor data at a specific location.

        If `script` is None, it signals that the current timepoint is done.

        Args:
            script (Script or None): The script object to execute, or None to signal end of timepoint.
            location (str): The location associated with the sensor data for the script.
        """
        # Conditional Logic: Appends the script and location to the list if a script is provided.
        if script is not None:
            self.scripts.append((script, location))
            # Functional Utility: Sets the event to notify the DeviceThread that a new script is available.
            self.script_received.set()
        else:
            # Functional Utility: Sets the event to signal the completion of scripts for the current timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        Acquires a lock for the specified location to ensure thread-safe access.

        Args:
            location (str): The location for which to retrieve sensor data.

        Returns:
            any: The sensor data at the specified location, or None if the location is invalid.
        """
        # Conditional Logic: Checks if the requested location exists in the sensor data.
        if location in self.sensor_data:
            # Functional Utility: Acquires the lock for the specific location to prevent concurrent access.
            self.locations_locks[location].acquire()
            return self.sensor_data[location]

        return None


    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.

        Releases the lock for the specified location after updating the data.

        Args:
            location (str): The location for which to set sensor data.
            data (any): The new sensor data value.
        """
        # Conditional Logic: Checks if the requested location exists in the sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Functional Utility: Releases the lock for the specific location after data update.
            self.locations_locks[location].release()


    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its dedicated thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the operational logic for a single Device instance.

    It continuously fetches neighbors, processes incoming scripts using a thread pool,
    and coordinates with other devices using a barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes a DeviceThread instance.

        Args:
            device (Device): The Device instance that this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Functional Utility: A thread pool to execute scripts concurrently.
        self.thread_pool = ThreadPool(8, device)

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Continuously gets neighbors, waits for scripts or timepoint completion,
        submits scripts to the thread pool, and synchronizes with other devices.
        """
        # Invariant: The device thread continuously processes operations.
        while True:
            # Functional Utility: Retrieves information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (e.g., system shutdown), breaks the loop.
            if neighbours is None:
                break

            # Invariant: Continuously checks for new scripts or timepoint completion.
            while True:

                # Conditional Logic: Checks if the timepoint is done and no new scripts are received yet.
                # If so, it clears the timepoint_done flag and sets script_received to prepare for the next cycle.
                if self.device.timepoint_done.wait() and not self.device.script_received.is_set():
                    self.device.timepoint_done.clear()
                    self.device.script_received.set()
                    break

                # Conditional Logic: If a script is received, it processes them.
                if self.device.script_received.is_set():
                    self.device.script_received.clear()

                    # Block Logic: Iterates through assigned scripts and submits them to the thread pool.
                    # Pre-condition: 'self.device.scripts' contains (script, location) tuples.
                    # Invariant: Each script will be submitted for execution.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task(script, location, neighbours)


            # Functional Utility: Waits for all tasks in the thread pool to complete before proceeding.
            self.thread_pool.tasks_queue.join()

            # Functional Utility: Synchronizes with other devices using the barrier before starting the next timepoint.
            self.device.barrier.wait()

        # Block Logic: After the main loop breaks, ensures all threads in the pool are joined.
        self.thread_pool.join_threads()


# Note: ThreadPool is defined below, potentially causing a circular import if it were in a separate file
# and imported at the top. Since it's in the same file, Python handles it.
class ThreadPool(object):
    """
    @brief Manages a pool of worker threads to execute scripts on sensor data concurrently.

    It uses a queue to hold tasks and distributes them among a fixed number of worker threads.
    """

    def __init__(self, number_threads, device):
        """
        @brief Initializes a ThreadPool instance.

        Args:
            number_threads (int): The number of worker threads in the pool.
            device (Device): The Device instance associated with this thread pool.
        """
        self.number_threads = number_threads
        # Functional Utility: List to hold the worker threads.
        self.device_threads = []
        self.device = device
        # Functional Utility: A queue to hold tasks (scripts, location, neighbors) for the worker threads.
        self.tasks_queue = Queue(number_threads)

        # Block Logic: Creates and appends worker threads to the pool.
        # Invariant: 'number_threads' worker threads will be created.
        for _ in range(0, number_threads): # Fixed xrange to range for Python 3 compatibility.
            thread = Thread(target=self.apply_scripts)
            self.device_threads.append(thread)


        # Block Logic: Starts all worker threads.
        # Invariant: All threads in 'device_threads' will be started.
        for thread in self.device_threads:
            thread.start()

    def apply_scripts(self):
        """
        @brief The main loop for a worker thread in the pool.

        Continuously retrieves tasks from the queue, executes the assigned script
        on the sensor data, and updates relevant device data.
        """
        # Invariant: Worker thread continuously processes tasks until a shutdown signal is received.
        while True:
            # Functional Utility: Retrieves a task (script, location, neighbors) from the queue.
            script, location, neighbours = self.tasks_queue.get()

            # Conditional Logic: Checks for a shutdown signal (None, None, None).
            # If received, marks the task as done and exits the thread.
            if neighbours is None and script is None:
                self.tasks_queue.task_done()
                return

            script_data = []
            # Block Logic: Collects sensor data from neighboring devices.
            # Pre-condition: 'neighbours' is a list of Device objects.
            # Invariant: Data from each neighbor (excluding self) at 'location' will be retrieved.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Functional Utility: Retrieves sensor data from the current device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Conditional Logic: If there is any script data, executes the script.
            if script_data != []:
                # Functional Utility: Executes the script with the collected sensor data.
                result = script.run(script_data)

                # Block Logic: Updates sensor data on neighboring devices with the script's result.
                # Pre-condition: 'neighbours' is a list of Device objects.
                # Invariant: Each neighbor (excluding self) will have its data at 'location' updated.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                # Functional Utility: Updates the current device's sensor data with the script's result.
                self.device.set_data(location, result)


            # Functional Utility: Marks the current task as complete in the queue.
            self.tasks_queue.task_done()


    def submit_task(self, script, location, neighbours):
        """
        @brief Submits a new task to the thread pool's queue for execution.

        Args:
            script (Script): The script to be executed.
            location (str): The location associated with the sensor data for the script.
            neighbours (list): A list of neighboring Device instances.
        """
        self.tasks_queue.put((script, location, neighbours))


    def join_threads(self):
        """
        @brief Signals the worker threads to shut down and waits for their completion.

        It does this by submitting 'None' tasks, which worker threads interpret as a shutdown signal.
        """
        # Functional Utility: Blocks until all currently enqueued tasks are processed.
        self.tasks_queue.join()

        # Block Logic: Submits shutdown signals to each worker thread.
        # Invariant: Each worker thread will receive a shutdown signal.
        for _ in range(0, len(self.device_threads)): # Fixed xrange to range for Python 3 compatibility.
            self.submit_task(None, None, None)

        # Block Logic: Waits for all worker threads to terminate.
        # Invariant: All threads in 'device_threads' will be joined.
        for thread in self.device_threads:
            thread.join()