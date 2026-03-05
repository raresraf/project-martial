"""
This module defines classes for simulating a distributed system, where individual
`Device` instances process sensor data and execute scripts in a multi-threaded
environment. It includes mechanisms for inter-device communication, synchronization,
and workload distribution among worker threads.
"""


from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device in the simulated distributed system.

    Each device has a unique ID, sensor data, and interacts with a supervisor.
    It manages scripts for processing, coordinates with other devices through
    a barrier, and uses worker threads for parallel execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor data for various locations.
            supervisor (object): The supervisor object responsible for coordinating devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.scripts_lock = Lock()
        self.timepoint_done = Event()
        self.barrier = None
        self.location_locks = {}
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources and synchronization mechanisms for a collection of devices.

        This static-like method initializes a reusable barrier for all devices
        and creates a global set of location-specific locks that all devices will share.

        Args:
            devices (list): A list of Device instances to be set up.
        """
        # Create a reusable barrier for all devices to synchronize at specific timepoints.
        # Pre-condition: 'devices' is a list of Device instances.
        barrier = ReusableBarrierCond(len(devices))
        # Assign the created barrier to each device.
        # Invariant: Each device in the list now shares the same barrier.
        for device in devices:
            device.barrier = barrier

        location_locks = {}

        # Initialize a lock for each unique location found across all devices' sensor data.
        # Invariant: 'location_locks' maps each unique location to a Lock object.
        for device in devices:
            for location in device.sensor_data:
                # If a lock for the current location does not exist, create one.
                if location not in location_locks:
                    location_locks[location] = Lock()

        # Assign the global set of location locks to each device.
        # Invariant: All devices share the same set of location-specific locks.
        for device in devices:
            device.location_locks = location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific location.

        If a script is provided, it's added to the device's script queue, and
        an event is set to signal that a new script has been received. If
        `script` is None, it signals that no more scripts are coming for the
        current timepoint.

        Args:
            script (object): The script object to assign, or None to signal
                             timepoint completion.
            location (str): The location associated with the script.
        """
        # Pre-condition: Check if a script is actually being assigned.
        if script is not None:
            # Acquire lock to safely add the script to the list.
            self.scripts_lock.acquire()
            self.scripts.append((script, location))
            self.scripts_lock.release()
            # Set event to notify the DeviceThread that a new script is available.
            self.script_received.set()
        else:
            # If script is None, it indicates that all scripts for the current
            # timepoint have been assigned.
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            any: The sensor data for the specified location, or None if the
                 location is not found in the device's sensor data.
        """
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (str): The location for which to set data.
            data (any): The new sensor data to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def process_work(self, script, location, neighbours):
        """
        Processes a script at a given location, incorporating data from neighbors.

        This method acquires a lock for the specific location to ensure exclusive
        access to its data during processing. It gathers data from the device's
        own sensors and its neighbors at the specified location, runs the script,
        and then updates the sensor data for both the device and its neighbors.

        Args:
            script (object): The script to execute.
            location (str): The location where the script is being processed.
            neighbours (list): A list of neighboring Device instances.
        """
        # Acquire a lock for the specific location to prevent race conditions when accessing/modifying data.
        self.location_locks[location].acquire()

        script_data = []

        # Block Logic: Gather data from neighboring devices at the specified location.
        # Invariant: 'script_data' collects all available data for the location from neighbors.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Gather data from the current device's own sensor at the specified location.
        data = self.get_data(location)
        if data is not None:
            script_data.append(data)

        # Pre-condition: Check if any data was collected to process.
        if script_data:
            # Execute the script with the collected data.
            result = script.run(script_data)

            # Block Logic: Update the sensor data of neighboring devices with the script's result.
            # Invariant: Each neighbor's data for the location is updated with the new result.
            for device in neighbours:
                device.set_data(location, result)

            # Update the current device's own sensor data with the script's result.
            self.set_data(location, result)

        # Release the lock for the location, allowing other threads to access it.
        self.location_locks[location].release()

    def shutdown(self):
        """
        Shuts down the device's worker thread.

        This method blocks until the device's associated `DeviceThread` has
        completed its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the primary execution logic for a Device in a separate thread.

    This thread is responsible for coordinating with a supervisor, managing
    a pool of worker threads, and processing scripts assigned to its device.
    It synchronizes with other `DeviceThread` instances using a barrier.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Executes the main logic of the device thread.

        This method sets up a pool of worker threads, continuously fetches
        scripts assigned to its device, distributes work to workers, and
        synchronizes with other device threads using a barrier. It also
        handles shutdown conditions.
        """
        work_lock = Lock()
        work_pool_empty = Event()
        work_pool_empty.set()  # Initially, the work pool is empty.
        work_pool = []  # Queue for scripts to be processed by workers.
        workers = []  # List of worker threads.
        workers_number = 7  # Fixed number of worker threads.
        work_available = Semaphore(0)  # Signals when work is available in the pool.
        own_work = None  # Stores a script to be processed directly by this thread.

        # Block Logic: Initializes and starts worker threads.
        # Invariant: Each worker is a separate thread ready to process scripts.
        for worker_id in range(1, workers_number + 1):
            workers.append(Worker(worker_id, work_pool, work_available, work_pool_empty, work_lock, self.device))
            workers[worker_id-1].start()

        # Main loop for the device thread, runs indefinitely until shutdown.
        while True:
            scripts_ran = []  # Tracks scripts processed in the current timepoint.
            
            # Retrieve information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: Check if the device has neighbors.
            if neighbours is not None:
                
                # Convert neighbors to a set for efficient manipulation and remove self if present.
                neighbours = set(neighbours)

                if self.device in neighbours:
                    neighbours.remove(self.device)

            # Pre-condition: Check if the device has no neighbors, indicating a shutdown scenario.
            if neighbours is None:
                # Release semaphores to unblock all worker threads.
                for i in range(0,7): # Invariant: Number of releases matches workers_number.
                    work_available.release()

                # Block Logic: Waits for all worker threads to complete and join.
                for worker in workers:
                    worker.join()
                break # Exit the main loop, effectively shutting down the device thread.

            # Synchronize all device threads before starting script processing for the current timepoint.
            self.device.barrier.wait()

            # Block Logic: Inner loop to process scripts for the current timepoint.
            # Invariant: Processes all assigned scripts for the timepoint before breaking.
            while True:
                # Wait until a new script is assigned or a timepoint done signal is received.
                self.device.script_received.wait()

                # Acquire lock to safely access the device's scripts list.
                self.device.scripts_lock.acquire()

                # Block Logic: Distributes assigned scripts.
                # Invariant: Each script is either processed directly or added to the work pool.
                for (script, location) in self.device.scripts:
                    
                    # Pre-condition: Check if the script has already been processed in this timepoint.
                    if script in scripts_ran:
                        continue # Skip already processed scripts.

                    
                    scripts_ran.append(script) # Mark script as processed for this timepoint.

                    # Block Logic: Assigns the script to either 'own_work' (for direct processing)
                    # or adds it to 'work_pool' for worker threads.
                    if own_work is None:
                        own_work = (script, location, neighbours)
                    
                    else:
                        work_lock.acquire() # Lock for work_pool access.
                        work_pool.append((script, location, neighbours))
                        work_pool_empty.clear() # Indicate work is available.
                        work_available.release() # Signal a worker.
                        work_lock.release() # Release work_pool lock.

                self.device.scripts_lock.release() # Release scripts list lock.

                # Pre-condition: All scripts for the current timepoint have been assigned and processed.
                if self.device.timepoint_done.is_set() and len(scripts_ran) == len(self.device.scripts):
                    
                    # Process the script assigned to 'own_work' if it exists.
                    if own_work is not None:
                        script, location, neighbours = own_work
                        own_work = None
                        self.device.process_work(script, location, neighbours)

                    # Wait until the work pool is empty (all workers have processed their tasks).
                    work_pool_empty.wait()

                    # Wait for all worker threads to signal they have completed their current work.
                    for worker in workers:
                        worker.work_done.wait()

                    self.device.timepoint_done.clear() # Reset timepoint done signal.
                    
                    # Synchronize all device threads again, marking the end of the timepoint.
                    self.device.barrier.wait()
                    break # Exit inner loop, signifying end of current timepoint processing.


class Worker(Thread):
    """
    Represents a worker thread that processes tasks from a shared work pool.

    Workers are managed by a `DeviceThread` and are responsible for executing
    scripts on behalf of the device. They synchronize access to the work pool
    and signal when their current task is complete.
    """

    def __init__(self, worker_id, work_pool, work_available, work_pool_empty, work_lock, device):
        """
        Initializes a Worker thread.

        Args:
            worker_id (int): A unique identifier for the worker.
            work_pool (list): The shared list of tasks to process.
            work_available (Semaphore): Semaphore to signal when work is available.
            work_pool_empty (Event): Event to signal when the work pool is empty.
            work_lock (Lock): Lock to protect access to the work pool.
            device (Device): The Device instance that owns this worker.
        """
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.work_pool = work_pool
        self.work_available = work_available
        self.work_pool_empty = work_pool_empty
        self.work_lock = work_lock
        self.device = device
        self.work_done = Event()
        self.work_done.set()

    def run(self):
        """
        Executes the main logic of the worker thread.

        The worker continuously waits for tasks to become available in the
        `work_pool`. Once a task is available, it acquires necessary locks,
        processes the script, and then signals its completion.
        """

        # Main loop for the worker thread, runs indefinitely until explicitly terminated.
        while True:
            # Acquire semaphore, blocking until a task is available in the work pool.
            self.work_available.acquire()
            # Acquire lock to safely access the work pool.
            self.work_lock.acquire()
            # Clear the 'work_done' event to signal that this worker is now busy.
            self.work_done.clear()

            # Pre-condition: Check if the work pool is empty after acquiring the lock.
            # Block Logic: If the work pool is empty, this worker's job is done; release
            # the lock and exit the thread. This handles shutdown.
            if not self.work_pool:
                self.work_lock.release()
                return # Exit the worker thread.

            # Pop the next task from the work pool. A task consists of a script, location, and neighbors.
            script, location, neighbours = self.work_pool.pop(0)

            # Pre-condition: Check if the work pool became empty after popping a task.
            # Block Logic: If the work pool is now empty, set the 'work_pool_empty' event
            # to signal to the DeviceThread.
            if not self.work_pool:
                self.work_pool_empty.set()

            self.work_lock.release() # Release work_pool lock.

            # Process the assigned script using the device's processing logic.
            self.device.process_work(script, location, neighbours)

            # Set the 'work_done' event to signal that this worker has completed its current task.
            self.work_done.set()
