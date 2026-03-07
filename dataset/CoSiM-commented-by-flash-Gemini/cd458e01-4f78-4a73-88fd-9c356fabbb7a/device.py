"""
@file device.py
@brief Implements components for a distributed system, likely a simulation or sensor network,
focusing on concurrent data processing and synchronization.
This module defines Device objects that manage sensor data and execute scripts.
It leverages a custom ThreadPool for parallel script execution and employs
location-specific semaphores and a reusable barrier for thread coordination.
"""

from threading import Thread, Semaphore, Condition
from pool_of_threads import ThreadPool # Assumed to be another local file, but is defined here later.

class Sem(object):
    """
    @brief Manages location-specific semaphores for controlling access to shared data locations.
    Ensures that only one thread can modify or read a specific location's data at a time.
    """

    def __init__(self, devices):
        """
        @brief Initializes the semaphore manager by creating a semaphore for each unique data location.

        @param devices (list): A list of all Device instances in the system.
        """
        self.location_semaphore = {}
        # Block Logic: Iterates through all devices and their sensor data to identify unique locations.
        for device in devices:
            for location in device.sensor_data:
                # Conditional Logic: Creates a new semaphore for a location if one doesn't already exist.
                if location not in self.location_semaphore:
                    self.location_semaphore[location] = Semaphore(value=1) # Binary semaphore (mutex).

    def acquire(self, location):
        """
        @brief Acquires the semaphore for the specified data location.
        Blocks if the semaphore is already acquired.

        @param location (int): The identifier of the data location.
        """
        self.location_semaphore[location].acquire()

    def release(self, location):
        """
        @brief Releases the semaphore for the specified data location.

        @param location (int): The identifier of the data location.
        """
        self.location_semaphore[location].release()


class ReusableBarrierCond(object):
    """
    @brief Implements a reusable barrier using a threading.Condition object.
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
    It processes assigned scripts using a dedicated thread and a ThreadPool.
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
        self.scripts = []  # List to hold (script, location) tuples assigned to this device.

        self.barrier = None  # Global barrier for device synchronization, assigned during setup.
        self.location_semaphore = None  # Manager for location-specific semaphores, assigned during setup.
        self.timepoint_done = False  # Flag indicating if script assignment for a timepoint is complete.

        self.thread = DeviceThread(self)  # The dedicated thread for this device's operations.
        self.thread.start()  # Starts the device's operational thread.

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return str: A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs global setup for devices, including initializing the global barrier and location semaphores.
        Device 0 coordinates this setup and propagates instances to other devices.

        @param devices (list): A list of all Device instances in the system.
        """
        # Conditional Logic: Only Device 0 performs the global setup.
        if self.device_id == 0:
            # Initializes the global barrier with the total number of devices.
            self.barrier = ReusableBarrierCond(len(devices))
            # Initializes the location semaphore manager.
            self.location_semaphore = Sem(devices)

            # Block Logic: Propagates the initialized barrier and semaphore manager to all other devices.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier
                    device.location_semaphore = self.location_semaphore

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location or signals timepoint completion.

        @param script (callable): The script (function or object with a run method) to execute.
                                  If None, it signals that script assignment for the timepoint is done.
        @param location (int): The identifier of the data location the script operates on.
        """
        # Conditional Logic: If a script is provided, it's added to the scripts list.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done = True  # Sets flag to indicate script assignment is complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location (int): The identifier of the data location.
        @return any: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

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
        @brief Shuts down the device's operational thread, waiting for its completion.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a Device.
    This thread manages the Device's lifecycle, including fetching neighbors,
    assigning scripts to a ThreadPool, and synchronizing with other devices.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device (Device): The Device object this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Initializes a ThreadPool for this device to execute scripts concurrently.
        self.thread_pool = ThreadPool(8, device)

    def run(self):
        """
        @brief The main execution loop of the DeviceThread.
        It continuously fetches neighbor information, waits for scripts to be assigned,
        submits them to its thread pool, waits for thread pool completion, and
        then synchronizes with other devices via a global barrier.
        """
        while True:
            # Retrieves neighbor devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (supervisor signals shutdown), terminates.
            if neighbours is None:
                break

            # Block Logic: Waits until scripts for the current timepoint are assigned (timepoint_done is True).
            while True:
                # Conditional Logic: Once scripts are assigned.
                if self.device.timepoint_done:
                    self.device.timepoint_done = False  # Resets the flag.
                    # Block Logic: Submits all assigned scripts to the thread pool for execution.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
                    break # Exits the inner loop to proceed with thread pool management.

            self.thread_pool.wait_threads()  # Waits for all submitted tasks in the thread pool to complete.

            self.device.barrier.wait()  # Synchronizes with all other devices using the global barrier.

        self.thread_pool.end_threads()  # Signals the thread pool to gracefully shut down its worker threads.


# This part defines the ThreadPool class within the same file, effectively self-contained.
# Previously, it was imported from 'pool_of_threads', suggesting a refactoring or organization choice.

class ThreadPool(object):
    """
    @brief Manages a pool of worker threads to execute tasks concurrently.
    Tasks (scripts) are submitted to a queue, and worker threads pick them up for processing.
    """
    
    def __init__(self, threads_count, device):
        """
        @brief Initializes the ThreadPool.

        @param threads_count (int): The number of worker threads to create in the pool.
        @param device (Device): The Device object associated with this thread pool.
        """
        self.queue = Queue(threads_count) # A queue to hold tasks for worker threads.
        self.threads = []  # List to hold the worker Thread objects.
        self.device = device # The associated Device instance.
        self.create_and_start_worker_threads(threads_count) # Creates and starts worker threads.

    def create_and_start_worker_threads(self, threads_count):
        """
        @brief Creates and starts the specified number of worker threads.
        Each worker thread continuously calls `do_job` to process tasks from the queue.

        @param threads_count (int): The number of worker threads to create.
        """
        # Block Logic: Creates worker threads targeting the `do_job` method.
        for _ in range(threads_count):
            thread = Thread(target=self.do_job)
            self.threads.append(thread)

        # Block Logic: Starts all created worker threads.
        for thread in self.threads:
            thread.start()

    def do_job(self):
        """
        @brief The main loop for a worker thread.
        It continuously retrieves tasks from the queue, executes the script,
        and updates data on devices, ensuring proper synchronization.
        """
        while True:
            # Retrieves a task (neighbours, script, location) from the queue.
            # Blocks if the queue is empty until a task is available.
            neighbours, script, location = self.queue.get()

            # Conditional Logic: Termination signal for worker threads.
            # If all components of the task are None, it indicates a shutdown signal.
            if neighbours is None and script is None and location is None:
                self.queue.task_done() # Signals that this termination task is done.
                return # Terminates the worker thread.

            # Block Logic: Processes the script with collected data.
            script_data = []
            # Acquires the location-specific semaphore before accessing shared data.
            self.device.location_semaphore.acquire(location)
            
            # Block Logic: Collects data from neighboring devices.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Collects data from its own device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Conditional Logic: If any data was collected, executes the script and propagates results.
            if script_data != []:
                
                result = script.run(script_data) # Executes the script.

                # Block Logic: Propagates the new data to all neighboring devices.
                for device in neighbours:
                    device.set_data(location, result)

                # Updates data on its own device.
                self.device.set_data(location, result)
            self.device.location_semaphore.release(location) # Releases the location-specific semaphore.

            self.queue.task_done() # Signals that the current task is complete in the queue.

    def submit(self, neighbours, script, location):
        """
        @brief Submits a new task (script with its context) to the thread pool queue.

        @param neighbours (list): A list of neighboring Device objects.
        @param script (callable): The script to execute.
        @param location (int): The data location the script operates on.
        """
        self.queue.put((neighbours, script, location))

    def wait_threads(self):
        """
        @brief Blocks until all tasks currently in the queue have been processed.
        """
        self.queue.join()

    def end_threads(self):
        """
        @brief Signals all worker threads to terminate and waits for their completion.
        """
        # Block Logic: Submits a termination signal for each worker thread.
        for _ in range(len(self.threads)):
            self.submit(None, None, None)

        self.wait_threads() # Waits for all termination signals to be processed.

        # Block Logic: Joins all worker threads to ensure they have fully shut down.
        for thread in self.threads:
            thread.join()
