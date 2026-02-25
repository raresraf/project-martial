


from Queue import Queue
from threading import Thread

class Worker(Thread):
    """
    A worker thread that continuously retrieves and processes tasks from a shared queue.

    Each worker is designed to execute scripts for a specific device, handling
    data collection from neighbors, script execution, and updating sensor data
    while respecting location-specific locks.
    """
    
    def __init__(self, tasks, device):
        """
        Initializes a new Worker thread.

        Args:
            tasks (Queue): The shared queue from which tasks are retrieved.
            device (Device): The Device object for which this worker will execute scripts.
        """
        Thread.__init__(self)
        self.tasks = tasks
        self.device = device
        self.daemon = True # Daemon threads exit automatically when the main program exits.
        self.start()       # Start the thread automatically upon creation.

    def run(self):
        """
        The main execution loop for the Worker thread.

        It continuously fetches tasks (neighbours, script, location) from the `tasks` queue.
        If a `None` sentinel is received, the worker terminates. Otherwise, it acquires
        a lock for the specific location, executes the script via `_script`, and
        marks the task as done.
        """
        while True:
            # Retrieve a task from the queue. This call blocks until an item is available.
            neighbours, script, location = self.tasks.get()

            # Check for the sentinel value (None) to signal thread termination.
            if neighbours is None:
                self.tasks.task_done() # Mark task as done before exiting.
                break                  # Terminate the thread.
            
            # Acquire the lock for the specific location to ensure exclusive access during script execution.
            with self.device.locations_locks[location]:
                self._script(neighbours, script, location) # Execute the script.
            
            self.tasks.task_done() # Mark the task as done in the queue.

    def _script(self, neighbours, script, location):
        """
        Executes a single assigned script for a specific location.

        This method collects sensor data from the device and its neighbors for
        the given location, executes the script, and then updates the sensor data
        in both the current device and its neighbors with the script's result.

        Args:
            neighbours (list): A list of neighboring Device objects for data collection.
            script (object): The script object to execute.
            location (int): The integer identifier of the location to process.
        """
        script_data = [] # List to store data collected for the script.
        # Collect data from neighboring devices for the current location.
        for neighbour in neighbours:
            data = neighbour.get_data(location)
            if data is not None:
                script_data.append(data)

        # Collect data from the current device itself for the current location.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # If any data was collected, execute the script.
        if script_data != []:
            # Execute the script with the collected data.
            result = script.run(script_data)

            # Update the data in neighboring devices.
            for neighbour in neighbours:
                neighbour.set_data(location, result)
            # Update the data in the current device.
            self.device.set_data(location, result)


class ThreadPool(object):
    """
    A simple thread pool implementation that manages a collection of `Worker` threads
    to execute tasks concurrently.

    Tasks are added to a shared queue, and worker threads retrieve and process them.
    The pool provides mechanisms to wait for task completion and to gracefully
    shut down all worker threads.
    """
    
    def __init__(self, num_threads):
        """
        Initializes a new ThreadPool.

        Args:
            num_threads (int): The initial capacity of the task queue.
                                The actual number of worker threads is set later.
        """
        # A queue to hold tasks for the worker threads. Its size is initialized here.
        self.tasks = Queue(num_threads)
        # List to store references to the worker threads.
        self.threads = []
        # Reference to the Device object that the workers in this pool will operate on.
        self.device = None

    def set_device(self, device, num_threads):
        """
        Assigns a `Device` to the thread pool and creates the specified number
        of `Worker` threads, associating them with this device.

        This method effectively starts the worker threads.

        Args:
            device (Device): The Device object associated with this thread pool.
            num_threads (int): The number of worker threads to create.
        """
        self.device = device
        # Create and start worker threads.
        for _ in range(num_threads):
            self.threads.append(Worker(self.tasks, self.device))

    def add_tasks(self, neighbours, script, location):
        """
        Adds a new task to the thread pool's queue.

        A task consists of the neighbors relevant to the script, the script itself,
        and the location it pertains to.

        Args:
            neighbours (list): A list of neighboring Device objects.
            script (object): The script object to be executed.
            location (int): The location identifier associated with the script.
        """
        self.tasks.put((neighbours, script, location))

    def wait_completion(self):
        """
        Blocks until all tasks previously added to the queue have been processed.
        """
        self.tasks.join()

    def end_threads(self):
        """
        Gracefully terminates all worker threads in the pool.

        It first waits for any remaining tasks to complete, then adds a sentinel
        (None) task for each worker to signal its termination, and finally
        joins all worker threads to ensure they have exited.
        """
        # Wait for all current tasks in the queue to be processed.
        self.tasks.join()
        # Add a None sentinel for each worker thread to signal it to stop.
        for _ in range(len(self.threads)):
            self.add_tasks(None, None, None)

        # Wait for all worker threads to actually terminate.
        for thread in self.threads:
            thread.join()





from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
from ThreadPool import ThreadPool

class Device(object):
    """
    Represents a simulated device within a multi-device system.

    Each device manages its own sensor data, processes assigned scripts,
    and coordinates with other devices through a shared barrier and
    location-specific locks for data consistency.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor readings,
                                 where keys are locations and values are data.
            supervisor (Supervisor): A reference to the supervisor object
                                     that manages overall simulation and device interactions.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when new scripts have been received for the current timepoint.
        self.script_received = Event()
        # List of scripts assigned to this device for current processing round.
        self.scripts = []
        # Event to signal that all scripts for the current timepoint are ready.
        self.timepoint_done = Event()

        # Reference to the shared reusable barrier for inter-device synchronization.
        self.barrier = None
        # Dictionary of Locks, one for each location in sensor_data, to protect data access.
        self.locations_locks = {}
        # List to store references to all devices in the system, populated during setup.
        self.devices = []
        # Initialize locks for each location present in the initial sensor data.
        for location in sensor_data:
            self.locations_locks[location] = Lock()

        # The dedicated thread for this device's main operational logic.
        self.thread = DeviceThread(self)
        # Start the device's operational thread upon initialization.
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
        Configures shared resources, specifically the synchronization barrier
        and location-specific locks, for all devices in the system.

        A new `ReusableBarrierSem` is created and shared among all devices.
        This device's `locations_locks` dictionary is also shared with other devices.

        Args:
            devices (list): A list of all `Device` instances in the system.
        """
        # Create a new reusable barrier for synchronizing all devices.
        barrier = ReusableBarrierSem(len(devices))
        self.barrier = barrier
        # Distribute the shared barrier and the locations_locks to all other devices.
        for dev in devices:
            dev.barrier = barrier
            dev.locations_locks = self.locations_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device.

        If a script is provided, it's added to the device's list of scripts.
        It also ensures that a lock exists for the given location.
        If `script` is None, it signals that script assignment for the current
        timepoint is complete.

        Args:
            script (object or None): The script object to assign, or None to signal
                                     the end of script assignments for a timepoint.
            location (int): The identifier for the location associated with the script.
        """
        if script is not None:
            # Add the script and its location to the device's scripts list.
            self.scripts.append((script, location))
            # Ensure a lock exists for this location (if not already initialized).
            if location not in self.locations_locks:
                self.locations_locks[location] = Lock()
        else:
            # If script is None, signal that all scripts for this timepoint have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Note: The locking for data access is handled externally by the `Worker` threads
              within the `ThreadPool` using `locations_locks`.

        Args:
            location (int): The identifier for the location for which to retrieve data.

        Returns:
            Any: The sensor data if the location exists in sensor_data, otherwise None.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Note: The locking for data access is handled externally by the `Worker` threads
              within the `ThreadPool` using `locations_locks`.

        Args:
            location (int): The identifier for the location where the data should be set.
            data (Any): The new data to set for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its associated main thread.
        This ensures that the device's main thread completes its execution.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    Manages the overall timestep progression and coordinates script processing
    for a Device using a `ThreadPool`.

    This thread continuously fetches neighbor information, waits for scripts
    to be assigned, dispatches these scripts to the `ThreadPool` for concurrent
    execution, waits for their completion, and then synchronizes with other
    DeviceThreads using a shared barrier. It also handles the graceful
    shutdown of the `ThreadPool`.
    """
    

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Initialize a ThreadPool with 8 worker threads.
        self.thread_pool = ThreadPool(8)

    def run(self):
        """
        The main execution loop for the DeviceThread.

        Each iteration represents a processing round. It performs the following steps:
        1. Sets up the `ThreadPool` with the current device.
        2. Retrieves updated neighbor information from the supervisor.
        3. If no neighbors are returned (e.g., simulation end), the loop breaks.
        4. Waits until scripts for the current timepoint have been assigned.
        5. Adds all assigned scripts to the `ThreadPool` for processing.
        6. Clears the `script_received` event for the next round.
        7. Waits for all tasks in the `ThreadPool` to complete.
        8. Synchronizes with other DeviceThreads using the shared `barrier`.
        9. Upon loop termination, it calls `thread_pool.end_threads()` to
           gracefully shut down its worker threads.
        """
        # Set the device for the thread pool and create worker threads.
        self.thread_pool.set_device(self.device, 8)

        while True:
            # Retrieve updated neighbor information from the supervisor for the current round.
            neighbours = self.device.supervisor.get_neighbours()
            # If supervisor returns None, it signals the simulation to terminate.
            if neighbours is None:
                break
            # Wait until scripts for the current timepoint have been assigned.
            self.device.script_received .wait()
            # Add all assigned scripts to the thread pool as tasks.
            for (script, location) in self.device.scripts:
                # The arguments for add_tasks are (neighbours, script, location) based on ThreadPool.add_tasks signature
                self.thread_pool.add_tasks(neighbours, script, location)

            # Clear the event, indicating that scripts for this round have been dispatched.
            self.device.script_received .clear()
            # Wait for all tasks added to the thread pool in this round to complete.
            self.thread_pool.wait_completion()
            # Wait at the shared barrier to synchronize with all other devices.
            self.device.barrier.wait()

        # After the main loop breaks (simulation termination), gracefully shut down the thread pool.
        self.thread_pool.end_threads()
