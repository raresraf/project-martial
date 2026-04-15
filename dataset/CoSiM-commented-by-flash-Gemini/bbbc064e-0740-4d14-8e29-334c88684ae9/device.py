

"""
@bbbc064e-0740-4d14-8e29-334c88684ae9/device.py
@brief Defines the Device, WorkerThread, and ReusableBarrier classes for managing distributed device operations.
This module provides the core components for a distributed simulation or data processing system.
The `Device` class represents an individual processing unit, holding sensor data and
managing scripts, while `DeviceThread` handles the execution logic for each device,
including task queuing and synchronization.

Domain: Concurrency, Distributed Systems, Simulation, Data Processing.
"""

from Queue import Queue
from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.
    This barrier allows a fixed number of threads to wait for each other at a
    specific point in their execution, and then proceeds together. It can be
    reused multiple times after all threads have passed through.
    Algorithm: Double-counting semaphore-based barrier.
    """
    

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier for a specified number of threads.
        @param num_threads: The total number of threads that will participate in the barrier.
        Functional Utility: Sets up internal counters and semaphores required for barrier synchronization.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Inline: Counter for the first phase of the barrier.
        self.count_threads2 = [self.num_threads] # Inline: Counter for the second phase of the barrier.
        self.count_lock = Lock()                 # Inline: A lock to protect access to the thread counters.
        self.threads_sem1 = Semaphore(0)         # Inline: Semaphore for releasing threads in the first phase.
        self.threads_sem2 = Semaphore(0)         # Inline: Semaphore for releasing threads in the second phase.

    def wait(self):
        """
        @brief Blocks the calling thread until all other threads have also called wait.
        Functional Utility: Orchestrates the two-phase synchronization mechanism, ensuring
        all participating threads reach this point before any can proceed.
        """
        # Block Logic: Executes the first phase of the barrier.
        self.phase(self.count_threads1, self.threads_sem1)
        # Block Logic: Executes the second phase of the barrier.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements one phase of the barrier synchronization.
        @param count_threads: A list containing the current count of threads waiting for this phase.
        @param threads_sem: The semaphore used to release threads once the count reaches zero.
        Block Logic: Decrements the thread count. When the count reaches zero,
        all waiting threads are released, and the count is reset for the next use.
        Invariant: At the entry, threads are waiting for their turn in the phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Block Logic: Checks if this is the last thread to reach the barrier phase.
            if count_threads[0] == 0:
                # Block Logic: Releases all waiting threads from the semaphore and resets the counter.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads # Inline: Resets the counter for the next use of the barrier.
        # Functional Utility: Acquires the semaphore, blocking the thread until all threads have reached the barrier.
        threads_sem.acquire()


class Device(object):
    """
    @brief Represents a single device in a distributed system, managing its sensor data and scripts.
    This class encapsulates device-specific state, including its ID, sensor readings,
    and a list of scripts to be executed. It interacts with a supervisor for global
    context and uses a barrier for synchronization with other devices. It also manages
    a pool of worker threads to process tasks.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: Unique identifier for the device.
        @param sensor_data: A dictionary containing sensor readings, indexed by location.
        @param supervisor: A reference to the supervisor object managing all devices.
        Functional Utility: Sets up the device's state, including synchronization primitives,
        a task queue, a pool of worker threads, and a dedicated thread for managing timepoints.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []                      # Inline: List to hold (script, location) tuples assigned to the device.
        self.timepoint_done = Event()          # Inline: Event to signal the completion of a timepoint's processing.

        # Functional Utility: Synchronization barrier for coordinating with other devices.
        # This will be assigned by the master device during setup.
        # This barrier is used for global synchronization across all devices.
        self.barrier = None
        self.queue = Queue()                   # Inline: A task queue for worker threads to pull scripts from.
        # Functional Utility: Initializes a pool of WorkerThreads for this device.
        self.workers = [WorkerThread(self) for _ in range(8)]

        # Functional Utility: Initializes and starts the dedicated thread for this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Block Logic: Starts all worker threads, making them ready to process tasks from the queue.
        for thread in self.workers:
            thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        Functional Utility: Provides a human-readable identifier for the device.
        """
        # Functional Utility: Formats the device ID into a descriptive string.
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the synchronization barrier and shared locks across multiple devices.
        @param devices: A list of all participating Device objects.
        Block Logic: If this is the master device (device_id == 0), it initializes a
        ReusableBarrier and a set of shared locks for data locations. It then assigns these
        synchronization primitives to all participating devices.
        Pre-condition: Called once all Device objects have been instantiated.
        """
        # Block Logic: Checks if the current device is the master device (ID 0).
        if self.device_id == 0:

            # Functional Utility: Initializes a ReusableBarrier with the total number of devices.
            barrier = ReusableBarrier(len(devices))

            # Functional Utility: Initializes a dictionary to hold locks for each unique data location across all devices.
            locks = {}

            # Block Logic: Populates the shared locks dictionary with a Lock for each unique data location.
            # This ensures that access to sensor data is thread-safe across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if not location in locks:
                        locks[location] = Lock()

            # Block Logic: Assigns the common barrier and shared locks to all participating devices.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location on the device.
        @param script: The script object to execute.
        @param location: The data location on which the script will operate.
        Block Logic: Appends the script and its target location to the device's script list.
        If no script is provided, it signals that the current timepoint is done.
        """
        # Block Logic: Checks if a script has been provided.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set() # Inline: Signals that the current timepoint has completed all script assignments.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key identifying the sensor data to retrieve.
        Returns: The sensor data at the specified location, or None if the location is not found.
        """
        # Block Logic: Checks if the requested location exists in the sensor data.
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a given location.
        @param location: The key identifying the sensor data to update.
        @param data: The new data value to set.
        Functional Utility: Writes new data to a specific location.
        """
        # Block Logic: Checks if the requested location exists in the sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Joins the device's main processing thread, effectively stopping its operation.
        Functional Utility: Ensures that the DeviceThread completes its execution before the program exits.
        """
        self.thread.join()


class WorkerThread(Thread):
    """
    @brief A worker thread responsible for executing assigned scripts on a device's data.
    Each worker thread continuously fetches tasks (script and location) from the device's queue,
    acquires a lock for the data location, collects data from neighbors and itself,
    executes the script, and propagates the results.
    """
    

    def __init__(self, device):
        """
        @brief Initializes the WorkerThread with its associated device.
        @param device: The Device object that this thread will serve.
        Functional Utility: Associates the worker thread with a specific device to access its resources.
        """
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the WorkerThread.
        Functional Utility: Continuously processes tasks from the device's queue.
        It handles data retrieval, script execution, and result propagation,
        ensuring thread-safe access to data locations.
        """
        while True:
            # Block Logic: Retrieves a task (script and location) from the device's queue.
            # Invariant: If `item` is None, it signifies a termination signal for this worker thread.
            item = self.device.queue.get()
            if item is None:
                break

            (script, location) = item

            # Block Logic: Acquires a lock for the specific data location to ensure exclusive access during processing.
            with self.device.locks[location]:
                script_data = []

                # Block Logic: Collects data from neighboring devices.
                # It iterates through all neighbors and retrieves data for the specified location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Block Logic: Collects data from the current device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if there is any data to process.
                # Pre-condition: `script_data` must not be empty for script execution.
                if script_data != []:
                    # Functional Utility: Runs the assigned script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Propagates the script result to neighboring devices.
                    # It iterates through all neighbors and updates their data for the specified location.
                    for device in self.device.neighbours:
                        device.set_data(location, result)

                    # Functional Utility: Updates the current device's data with the script result.
                    self.device.set_data(location, result)

            # Functional Utility: Marks the current task as done in the device's queue.
            self.device.queue.task_done()


class DeviceThread(Thread):
    """
    @brief A dedicated thread for a Device to manage its timepoints and orchestrate task distribution.
    This thread coordinates the overall process for a single device, fetching neighbor information,
    signaling timepoint completion, distributing scripts to worker threads,
    and synchronizing with other devices using a barrier.
    """
    

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread with its associated device.
        @param device: The Device object that this thread will manage.
        Functional Utility: Sets up the thread name and associates it with the device.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Orchestrates the processing at each timepoint for the device.
        It manages the flow of execution, including fetching data, queuing scripts,
        and ensuring synchronization with other devices.
        """
        while True:

            # Functional Utility: Retrieves the current set of neighboring devices from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Checks for a termination signal from the supervisor.
            # If neighbors are None, it signals that the simulation or processing should stop.
            if self.device.neighbours is None:
                break

            # Functional Utility: Waits until all scripts for the current timepoint have been assigned
            # and signals that the timepoint is done.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Inline: Resets the event for the next timepoint.

            # Block Logic: Distributes assigned scripts to the device's worker queue.
            # Invariant: Each script (with its location) is placed into the queue for a worker to pick up.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location))

            # Functional Utility: Waits for all tasks (scripts) put into the device's queue to be processed by worker threads.
            self.device.queue.join()

            # Functional Utility: Synchronizes with other device threads using the shared barrier,
            # ensuring all devices complete their current timepoint before proceeding.
            self.device.barrier.wait()

        # Block Logic: Inserts termination signals into the device's queue for each worker thread.
        # This ensures all worker threads exit gracefully.
        for _ in range(8): # Inline: Assuming 8 worker threads as initialized in Device.__init__.
            self.device.queue.put(None)

        # Block Logic: Waits for all worker threads to complete their execution (after processing termination signals).
        for thread in self.device.workers:
            thread.join()
