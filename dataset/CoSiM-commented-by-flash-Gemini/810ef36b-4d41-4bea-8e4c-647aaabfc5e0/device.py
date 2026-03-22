


import multiprocessing
from threading import Event, Thread, Lock
from threadpool import ThreadPool
from reusablebarrier import ReusableBarrier


class Device(object):
    """
    Represents a single computational device in a simulated distributed system.
    Each device has an ID, manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts using a `ThreadPool` of worker threads.
    It synchronizes with other devices using a shared barrier and per-location locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device object.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing sensor readings or local data,
                                 keyed by location.
            supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to store (script, location) tuples assigned to this device.
        # Event to signal when all scripts for the current timepoint have been assigned.
        self.time_point_done = Event()
        # Number of CPU cores, used to determine the size of the thread pool.
        self.nr_cpu = multiprocessing.cpu_count()
        # The main thread for this device, which handles its operational lifecycle and thread pool.
        self.thread = DeviceThread(self, self.nr_cpu)

        # Dictionary of locks, where each lock protects data at a specific location.
        # This will be shared across all devices for per-location data access control.
        self.locations_lock_set = None
        # Barrier for synchronization with other devices. Initialized to None,
        # will be properly configured by setup_devices.
        self.barrier = None

        self.thread.start() # Start the device's operational thread.

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the shared barrier and per-location locks for all devices.
        Only device with ID 0 initializes these shared resources and distributes them.

        Args:
            devices (List[Device]): A list of all Device instances in the system.
        """
        # Block Logic: Only the device with ID 0 initializes the shared resources (barrier, locks)
        # and then distributes them to all other devices.
        if self.device_id == 0:
            # Initializes a ReusableBarrier with the total number of devices.
            my_barrier = ReusableBarrier(len(devices))

            # Block Logic: All devices get a reference to this shared barrier.
            for dev in devices:
                dev.barrier = my_barrier

            # Dictionary to store per-location locks.
            locations_lock_set = {}

            # Block Logic: Iterates through all devices and their sensor data to collect
            # all unique location keys and create a Lock for each.
            for dev in devices:
                for location in dev.sensor_data:
                    if location not in locations_lock_set:
                        locations_lock_set[location] = Lock()

            # Block Logic: All devices get a reference to this shared set of location locks.
            for dev in devices:
                dev.locations_lock_set = locations_lock_set

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a specific location.
        If script is None, it signals that the timepoint's script assignments are done.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The logical location (index) associated with the script's execution.
        """
        # Conditional Logic: If a script object is provided, add it to the device's script list.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If script is None, it means no more scripts for this timepoint,
            # so signal completion for the script assignment phase.
            self.time_point_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The key (index) for the sensor data.

        Returns:
            Any: The data associated with the location, or None if not found.
        """
        # Conditional Logic: Checks if the location exists in sensor_data before accessing.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (int): The key (index) for the sensor data.
            data (Any): The new data to set for the location.
        """
        # Conditional Logic: Updates data only if the location already exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its operational thread, ensuring all
        tasks are completed before the program exits.
        """
        self.thread.join()



class DeviceThread(Thread):
    """
    The main operational thread for a Device. It drives the device's behavior,
    synchronizing with other devices via a barrier and distributing assigned scripts
    to a `ThreadPool` of worker threads for parallel execution.
    """
    

    def __init__(self, device, nr_cpu):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The Device instance that this thread controls.
            nr_cpu (int): The number of CPU cores, determining the size of the thread pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.nr_cpu = nr_cpu
        # Initializes a ThreadPool for executing scripts with `nr_cpu` workers.
        self.pool = ThreadPool(nr_cpu, device)

    def run(self):
        """
        The main execution loop for the DeviceThread.
        It repeatedly fetches neighbors, waits for script assignments,
        distributes scripts to the `ThreadPool`, waits for their completion,
        and then synchronizes with other devices using a barrier.
        """
        while True:
            # Fetches the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Conditional Logic: If no neighbors are returned (e.g., system shutdown),
            # send termination signals to the thread pool workers and exit the loop.
            if neighbours is None:
                # Block Logic: Add `nr_cpu` termination tasks to the queue to signal workers to exit.
                for _ in xrange(self.nr_cpu):
                    self.pool.add_task((True, None, None))

                # Block Logic: Wait for all worker threads to acknowledge termination and join.
                self.pool.wait_workers()
                break

            # Block Logic: Waits for the supervisor to signal that all scripts
            # for the current timepoint have been assigned.
            self.device.time_point_done.wait()

            # Clears the event for the next timepoint.
            self.device.time_point_done.clear()

            # Block Logic: Adds each assigned script as a task to the thread pool.
            for my_script in self.device.scripts:
                self.pool.add_task((False, my_script, neighbours))

            # Block Logic: Waits for all tasks currently in the thread pool to complete.
            self.pool.wait_completion()

            # Block Logic: Synchronizes with other DeviceThreads at a shared barrier.
            # This ensures all devices complete their script processing before proceeding.
            self.device.barrier.wait()


from threading import Thread
from Queue import Queue


class ThreadPool(object):
    """
    A generic thread pool implementation that manages a fixed number of worker threads
    (`AuxiliaryDeviceThread`) to execute tasks from a queue.
    """
    

    def __init__(self, num_threads, device):
        """
        Initializes a ThreadPool.

        Args:
            num_threads (int): The number of worker threads in the pool.
            device (Device): The parent Device instance that this pool serves.
        """
        self.queue = Queue(num_threads) # A queue to hold tasks for the workers.
        self.device = device
        self.workers = [] # List to hold AuxiliaryDeviceThread instances.
        # Block Logic: Creates and stores worker threads.
        for _ in xrange(num_threads):
            adt = AuxiliaryDeviceThread(self.device, self.queue)
            self.workers.append(adt)

    def add_task(self, info):
        """
        Adds a new task to the thread pool's queue.

        Args:
            info (Tuple): A tuple containing task information (e.g., (can_finish, script, neighbours)).
        """
        self.queue.put(info)

    def wait_completion(self):
        """
        Blocks until all tasks currently in the queue have been processed.
        """
        self.queue.join()

    def wait_workers(self):
        """
        Blocks until all worker threads in the pool have terminated.
        This is typically called during shutdown.
        """
        for adt in self.workers:
            adt.join()


class AuxiliaryDeviceThread(Thread):
    """
    A worker thread within the ThreadPool. It continuously fetches tasks from
    the shared queue, executes scripts (gathering data from neighbors and its
    parent device), and updates data, ensuring thread safety with location-specific locks.
    """
    

    def __init__(self, device, queue):
        """
        Initializes an AuxiliaryDeviceThread.

        Args:
            device (Device): The parent Device instance whose data and locks it will access.
            queue (Queue): The shared task queue from which to retrieve tasks.
        """
        Thread.__init__(self)
        self.queue = queue
        self.device = device
        self.daemon = True # Set as daemon so it terminates with the main program.
        self.start()

    def run(self):
        """
        The main execution loop for the AuxiliaryDeviceThread.
        It continuously retrieves tasks from the queue, processes them,
        and marks them as done. It exits when it receives a termination signal.
        """
        while True:
            # Gets a task from the queue. This call blocks until an item is available.
            can_finish, got_script, neighbours = self.queue.get()

            # Conditional Logic: If `can_finish` is True, it's a termination signal.
            if can_finish:  
                # Mark the termination task as done.
                self.queue.task_done()
                break # Exit the worker thread's loop.

            script, location = got_script # Unpack the script and location from the task.
            
            # Block Logic: Acquire the location-specific lock to protect data access for this location.
            self.device.locations_lock_set[location].acquire()

            script_data = [] # Data collected for the script.

            # Block Logic: Collects data from neighboring devices for the specified location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: Collects data from the current device for the specified location.
            data = self.device.get_data(location)


            if data is not None:
                script_data.append(data)

            # Conditional Logic: If there is any data, execute the script.
            if script_data: # Equivalent to if script_data is not empty.
                # Executes the script with the collected data.
                result = script.run(script_data)

                # Block Logic: Updates the data on neighboring devices with the script's result.
                for device in neighbours:
                    device.set_data(location, result)

                # Block Logic: Updates the data on the current device with the script's result.
                self.device.set_data(location, result)

            # Release the location-specific lock.
            self.device.locations_lock_set[location].release()

            # Mark the current task as done in the queue.
            self.queue.task_done()
