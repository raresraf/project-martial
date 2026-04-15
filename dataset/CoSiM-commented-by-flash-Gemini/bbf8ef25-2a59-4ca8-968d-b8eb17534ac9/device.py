

from threading import Semaphore, Lock, Event, Thread
from Queue import Queue
from barrier import ReusableBarrier

class Device(object):
    """
    @brief Represents an individual device in a distributed system, managing sensor data and script tasks.
    This class handles device-specific data, communicates with a central supervisor,
    and coordinates its operations (script execution, data access) with other devices
    using a shared barrier and semaphores for location-specific locking.
    """
    
    MAX_LOCATIONS = 100 # Inline: Defines the maximum number of distinct data locations.
    CONST_ONE = 1       # Inline: A constant used for initialization, representing a single unit or count.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance with its unique identifier, sensor data, and supervisor.
        @param device_id: Unique integer ID for this device.
        @param sensor_data: Dictionary of sensor readings, where keys are locations and values are data.
        @param supervisor: Reference to the central supervisor object managing all devices.
        Functional Utility: Sets up the device's basic state, including placeholders for shared
        synchronization primitives (barrier and semaphores) that are injected later by the master device.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []                      # Inline: List to hold (script, location) tuples assigned to the device.
        self.timepoint_done = Event()          # Inline: Event to signal the completion of a timepoint's processing.
        self.semaphore = []                    # Inline: List of shared Semaphores for location-specific data locking, initialized by the master device.
        self.thread = DeviceThread(self)       # Inline: The dedicated thread for this device's timepoint management.
        self.barrier = ReusableBarrier(Device.CONST_ONE) # Inline: Placeholder barrier, will be replaced by a shared instance.

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        Functional Utility: Provides a human-readable identifier for the device.
        """
        # Functional Utility: Formats the device ID into a descriptive string.
        return "Device %d" % self.device_id

    def get_distributed_objs(self, barrier, semaphore):
        """
        @brief Receives and sets the shared synchronization objects from the master device.
        @param barrier: The shared ReusableBarrier instance for global synchronization.
        @param semaphore: The shared list of Semaphores for location-specific data locking.
        Functional Utility: Integrates the device into the global synchronization and locking scheme.
        """
        
        self.barrier = barrier
        self.semaphore = semaphore
        self.thread.start() # Functional Utility: Starts the device's dedicated timepoint management thread after synchronization objects are set.

    def setup_devices(self, devices):
        """
        @brief Initializes and distributes shared synchronization objects if this is the master device.
        @param devices: A list of all Device objects in the system.
        Block Logic: If `device_id` is 0 (master device), it initializes a single ReusableBarrier
        for all devices and a list of Semaphores, one for each possible data location (up to MAX_LOCATIONS).
        It then distributes these objects to all other devices.
        Pre-condition: Called once all Device objects have been instantiated.
        """
        
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            semaphore = []
            i = Device.MAX_LOCATIONS
            # Block Logic: Initializes a list of semaphores, one for each potential data location.
            # Each semaphore has an initial value of 1, allowing one thread to acquire it at a time.
            while i > 0:
                semaphore.append(Semaphore(value=Device.CONST_ONE))
                i = i - 1

            # Block Logic: Distributes the shared barrier and semaphore list to all devices.
            for device in devices:
                device.get_distributed_objs(barrier, semaphore)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be processed at a specific data location.
        @param script: The script object to be executed.
        @param location: The integer index representing the data location.
        Block Logic: Adds the script and its target location to the device's list of pending scripts.
        If no script is provided (i.e., `None`), it signals that the current timepoint's
        script assignment phase is complete.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set() # Inline: Signals that the current timepoint has completed all script assignments.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.
        @param location: The key (integer index) identifying the sensor data to retrieve.
        Returns: The sensor data object at the specified location, or None if not found.
        """
        
        if location in self.sensor_data:
            obj = self.sensor_data[location]
        else:
            obj = None
        return obj

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a specified location.
        @param location: The key (integer index) identifying the sensor data to update.
        @param data: The new data value to set at the location.
        Functional Utility: Modifies the device's local sensor data for a given location.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Joins the device's main processing thread, ensuring graceful termination.
        Functional Utility: Blocks until the `DeviceThread` associated with this device completes its execution.
        """
        
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the operational lifecycle of a Device, orchestrating script execution across timepoints.
    This thread is responsible for fetching neighbor information, signaling timepoint completion,
    distributing scripts to a pool of `WorkerThread`s via a queue, and synchronizing with
    other devices using a shared barrier.
    """
    
    THREADS_TO_START = 8  # Inline: Defines the number of worker threads to spawn for script execution.
    MAX_SCRIPTS = 100     # Inline: Defines the maximum number of scripts the queue can hold.

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread, associating it with a Device instance.
        @param device: The Device object that this thread will manage.
        Functional Utility: Sets up the thread's name, its association with the device,
        and initializes a queue for scripts to be processed by worker threads.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = []                      # Inline: List to hold references to the spawned WorkerThread instances.
        self.neighbours = []                   # Inline: List to store the current neighboring devices.
        self.queue = Queue(maxsize=DeviceThread.MAX_SCRIPTS) # Inline: Bounded queue for script tasks.

    def run(self):
        """
        @brief The core execution loop for the DeviceThread, managing timepoint-based processing.
        Functional Utility: Continuously fetches neighbor data, processes assigned scripts through
        a worker pool, and uses a barrier to synchronize with other devices, until a termination
        signal is received from the supervisor.
        """
        lock = Lock() # Inline: A local lock used to protect access to the script queue.
        # Block Logic: Spawns and starts a fixed number of WorkerThread instances.
        for i in range(DeviceThread.THREADS_TO_START):
            self.threads.append(WorkerThread(self, i, self.device, self.queue, lock))
            self.threads[i].setDaemon(True) # Inline: Sets worker threads as daemon so they terminate when the main program exits.
        for thread in self.threads:
            thread.start()

        while True:
            # Block Logic: Retrieves the current set of neighboring devices from the supervisor.
            # Invariant: `self.neighbours` will be None if the supervisor signals termination.
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break

            # Functional Utility: Waits until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Enqueues all scripts assigned to the device for the current timepoint.
            for script in self.device.scripts:
                self.queue.put(script)
            # Functional Utility: Blocks until all scripts currently in the queue have been processed by worker threads.
            self.queue.join()

            self.device.timepoint_done.clear() # Inline: Resets the event for the next timepoint.

            # Functional Utility: Synchronizes with other device threads using the shared barrier,
            # ensuring all devices complete their current timepoint before proceeding to the next.
            self.device.barrier.wait()

        # Block Logic: Sends termination signals to all worker threads by enqueuing `None`.
        for i in range(DeviceThread.THREADS_TO_START): # Inline: Sends a None signal for each worker thread to terminate.
            self.queue.put(None)
        
        # Block Logic: Waits for all worker threads to complete their execution (after processing termination signals).
        for thread in self.threads:
            thread.join()


class WorkerThread(Thread):
    """
    @brief Executes individual scripts fetched from a DeviceThread's queue.
    This thread is responsible for acquiring location-specific semaphores, collecting
    data from the local and neighboring devices, running the assigned script, and
    propagating the results back to the relevant devices.
    """
    

    def __init__(self, master, worker_id, device, queue, lock):
        """
        @brief Initializes a WorkerThread.
        @param master: Reference to the `DeviceThread` that spawned this worker.
        @param worker_id: Unique identifier for this worker thread.
        @param device: The `Device` object whose tasks this worker will execute.
        @param queue: The `Queue` from which to retrieve script tasks.
        @param lock: A shared `Lock` used to protect access to the task queue.
        Functional Utility: Sets up the worker with references to its parent `DeviceThread`,
        the associated `Device`, the task `Queue`, and the shared `Lock` for queue access.
        """
        
        Thread.__init__(self, name="Worker Thread %d %d" % (worker_id, device.device_id))
        self.master = master
        self.device = device
        self.queue = queue
        self.lock = lock

    def run(self):
        """
        @brief The main execution loop for the WorkerThread, processing scripts from the queue.
        Functional Utility: Continuously attempts to retrieve and execute script tasks.
        For each task, it acquires a location-specific semaphore, gathers data,
        runs the script, updates relevant devices, and then releases the semaphore,
        continuing until a termination signal is received.
        """

        while True:
            # Block Logic: Acquires a lock to safely check and retrieve an item from the queue.
            self.lock.acquire()
            value = self.queue.empty() # Inline: Checks if the queue is empty.
            if value is False:
                # Invariant: If the queue is not empty, a script and its location are dequeued.
                (script, location) = self.queue.get()
            self.lock.release() # Inline: Releases the lock after queue access.


            if value is False: # Block Logic: Proceeds with script execution if a task was retrieved.
                script_data = []

                # Functional Utility: Acquires a semaphore for the specific data location to ensure exclusive access during processing.
                self.device.semaphore[location].acquire()

                # Block Logic: Collects data from neighboring devices for the script.
                for device in self.master.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collects data from the current device for the script.
                data = self.master.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if there is any data to process.
                # Pre-condition: `script_data` must not be empty for script execution.
                if script_data != []:
                    # Functional Utility: Runs the assigned script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Propagates the script result to neighboring devices.
                    for device in self.master.neighbours:
                        device.set_data(location, result)
                    
                    # Functional Utility: Updates the current device's data with the script result.
                    self.master.device.set_data(location, result)

                self.device.semaphore[location].release() # Functional Utility: Releases the semaphore for the data location.
                self.queue.task_done()                    # Functional Utility: Marks the current task as done in the queue.

            # Block Logic: Checks if a termination signal has been received by the master DeviceThread.
            # If so, this worker thread also terminates.
            if self.master.neighbours is None:
                break
