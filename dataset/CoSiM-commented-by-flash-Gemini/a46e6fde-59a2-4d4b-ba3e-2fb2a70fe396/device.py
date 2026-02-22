"""
@a46e6fde-59a2-4d4b-ba3e-2fb2a70fe396/device.py
@brief Implements a simulated distributed device system with a reusable barrier for synchronization.

This module defines components for a network of 'Device' objects, each running in its own thread,
that process sensor data collaboratively. It includes a custom 'ReusableBarrier' for inter-device
synchronization and uses thread pools for concurrent script execution on sensor data.
"""

from threading import Semaphore, Lock, Thread, Event
import queue # Changed from Queue to queue for Python 3 compatibility

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.

    This barrier allows a specified number of threads to wait for each other at a
    synchronization point and then proceed together, and can be reused multiple times.
    It uses two semaphores to manage the two phases of a barrier.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes a ReusableBarrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier before any can proceed.
        """
        self.num_threads = num_threads
        # Functional Utility: Tracks the number of threads waiting in phase 1.
        self.count_threads1 = [self.num_threads]
        # Functional Utility: Tracks the number of threads waiting in phase 2.
        self.count_threads2 = [self.num_threads]
        # Functional Utility: A lock to protect access to the thread counters.
        self.count_lock = Lock()
        # Functional Utility: Semaphore for controlling threads in phase 1.
        self.threads_sem1 = Semaphore(0)
        # Functional Utility: Semaphore for controlling threads in phase 2.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all `num_threads` threads have arrived.

        This method coordinates two phases of waiting to make the barrier reusable.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the reusable barrier.

        Args:
            count_threads (list): A list containing a single integer, the count of remaining threads for this phase.
            threads_sem (Semaphore): The semaphore associated with this phase.
        """
        # Block Logic: Ensures exclusive access to the thread counter.
        with self.count_lock:
            count_threads[0] -= 1 # Functional Utility: Decrements the count of waiting threads.
            # Conditional Logic: If this is the last thread to arrive, releases all waiting threads.
            if count_threads[0] == 0:
                # Block Logic: Releases all threads that were waiting on this semaphore.
                # Pre-condition: 'count_threads[0]' is 0, meaning all threads have arrived.
                # Invariant: 'num_threads' release operations will be performed.
                for _ in range(self.num_threads): # Fixed xrange to range
                    threads_sem.release()
                count_threads[0] = self.num_threads # Functional Utility: Resets the counter for reuse.
        threads_sem.acquire() # Functional Utility: Acquires a permit from the semaphore, potentially blocking.


class Device(object):
    """
    @brief Represents a single simulated device in a distributed system.

    Manages its own sensor data, communicates with a supervisor, and processes
    scripts, utilizing a dedicated thread and shared synchronization mechanisms.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor data values.
            supervisor (Supervisor): An object responsible for overseeing devices.
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
        # Functional Utility: The dedicated thread for this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Functional Utility: Reference to the shared barrier for inter-device synchronization.
        self.barrier = None
        # Functional Utility: Dictionary of locks for different locations, shared across devices.
        self.locks = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (barrier and locks) for all devices.

        This method should typically be called once by a coordinating entity (e.g., device 0).

        Args:
            devices (list): A list of all Device instances in the system.
        """
        # Conditional Logic: Ensures setup is performed only once by device 0.
        if self.device_id == 0:
            # Functional Utility: Initializes the reusable barrier with the total number of devices.
            barrier = ReusableBarrier(len(devices))
            # Functional Utility: Initializes a dictionary to store shared locks for locations.
            locks = {}

            # Block Logic: Assigns the shared barrier and locks dictionary to all devices.
            # Pre-condition: 'devices' is a list of all Device objects.
            # Invariant: Each device will share the same barrier and locks dictionary.
            for i in range(len(devices)): # Fixed xrange to range
                devices[i].barrier = barrier
                devices[i].locks = locks

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
            self.script_received.set() # Functional Utility: Signals the DeviceThread that a script is ready.
        else:
            self.timepoint_done.set() # Functional Utility: Signals that processing for the current timepoint is complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        Args:
            location (str): The location for which to retrieve sensor data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        # Conditional Logic: Checks if the location exists in the device's sensor data.
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.

        Args:
            location (str): The location for which to set sensor data.
            data (any): The new sensor data value.
        """
        # Conditional Logic: Updates the sensor data if the location exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its dedicated thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the operational logic for a single Device instance.

    It continuously fetches neighbors, processes incoming scripts using an internal
    thread pool (implemented via a Queue and executor threads), and coordinates with
    other devices using a barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes a DeviceThread instance.

        Args:
            device (Device): The Device instance that this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Functional Utility: List to keep track of the executor threads.
        self.thread_list = []
        # Functional Utility: A queue to hold tasks (scripts, location, neighbors) for executor threads.
        self.queue = queue.Queue() # Changed from Queue.Queue to queue.Queue for Python 3 compatibility

        # Block Logic: Creates and starts a fixed number of executor threads.
        # Invariant: 8 executor threads will be created and started.
        for _ in range(8): # Fixed xrange to range
            thread = Thread(target=self.executor)
            thread.start()
            self.thread_list.append(thread)

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Continuously gets neighbors, waits for scripts or timepoint completion,
        submits scripts to its internal queue for execution, and synchronizes
        with other devices.
        """
        # Invariant: The device thread continuously processes data and scripts.
        while True:
            # Functional Utility: Retrieves information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Conditional Logic: If no neighbors are returned (e.g., system shutdown signal),
            # signals executor threads to shut down and breaks the loop.
            if neighbours is None:
                # Block Logic: Puts None into the queue multiple times to signal each executor thread to shut down.
                # Pre-condition: 'thread_list' contains active executor threads.
                # Invariant: Each executor thread receives a shutdown signal.
                for _ in range(8): # Fixed xrange to range
                    self.queue.put(None)
                self.shutdown() # Functional Utility: Waits for executor threads to join.
                self.thread_list = [] # Functional Utility: Clears the list of threads.
                break

            # Functional Utility: Waits for the timepoint_done event, indicating all scripts can be collected.
            self.device.timepoint_done.wait()

            # Block Logic: Puts each assigned script as a task into the queue for executor threads.
            # Pre-condition: 'self.device.scripts' contains (script, location) tuples.
            # Invariant: Each script will be submitted for execution.
            for (script, location) in self.device.scripts:
                self.queue.put((script, location, neighbours))

            # Functional Utility: Blocks until all tasks in the queue are processed by executor threads.
            self.queue.join()

            # Functional Utility: Synchronizes with other devices using the barrier before starting the next timepoint.
            self.device.barrier.wait()

            # Functional Utility: Clears the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()


    def executor(self):
        """
        @brief The main loop for an executor thread.

        Continuously retrieves tasks from the queue, executes the assigned script
        on the sensor data, and updates relevant device data, ensuring thread safety
        with locks per location.
        """
        # Invariant: Executor thread continuously processes tasks until a shutdown signal is received.
        while True:
            # Functional Utility: Retrieves a task from the queue, potentially blocking until a task is available.
            items = self.queue.get()

            # Conditional Logic: If a None item is received, it's a shutdown signal.
            if items is None:
                self.queue.task_done() # Functional Utility: Marks this task as done.
                break # Functional Utility: Exits the executor thread loop.

            script = items[0]
            location = items[1]
            neighbours = items[2]

            # Block Logic: Ensures a lock exists for the current location, and then acquires it.
            # Pre-condition: 'self.device.locks' is a shared dictionary of locks.
            # Invariant: Access to 'location' data is serialized by a dedicated lock.
            if location not in self.device.locks:
                self.device.locks[location] = Lock() # Functional Utility: Creates a new lock if one doesn't exist.

            self.device.locks[location].acquire() # Functional Utility: Acquires the lock for the current location.

            script_data = []
            # Block Logic: Collects sensor data from neighboring devices.
            # Pre-condition: 'neighbours' is a list of Device objects.
            # Invariant: Data from each neighbor at 'location' is collected.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Functional Utility: Collects sensor data from the current device itself.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Conditional Logic: If any script data was collected, executes the script.
            if script_data != []:
                # Functional Utility: Executes the script with the combined sensor data.
                result = script.run(script_data)

                # Block Logic: Updates sensor data on neighboring devices with the script's result.
                # Pre-condition: 'neighbours' is a list of Device objects.
                # Invariant: Each neighbor's data at 'location' is updated.
                for device in neighbours:
                    device.set_data(location, result)
                
                # Functional Utility: Updates the current device's sensor data with the script's result.
                self.device.set_data(location, result)

            # Functional Utility: Releases the lock for the current location after processing.
            self.device.locks[location].release()

            self.queue.task_done() # Functional Utility: Signals that this task has been completed.



    def shutdown(self):
        """
        @brief Joins all executor threads, waiting for their completion.
        """
        # Block Logic: Waits for each executor thread to finish execution.
        # Pre-condition: 'self.thread_list' contains all active executor threads.
        # Invariant: All threads in 'thread_list' will be joined.
        for i in range(8): # Fixed xrange to range
            self.thread_list[i].join()
