


"""
This module implements the core components for a device simulation framework.
It includes classes for individual devices (`Device`), their operational threads (`DeviceThread`),
and a mechanism for parallel script execution on sensor data (`WorkerPool`).
"""

from threading import Lock, Event, Thread
from barrier import ReusableBarrierCond
from workerpool import WorkerPool


class Device(object):
    """
    Represents a simulated device in the system, managing its sensor data,
    scripts, and interaction with a supervisor. Each device runs in its
    own thread to simulate concurrent operations.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor data, keyed by location.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        self.barrier = None
        self.locks = None

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
        Sets up shared resources (barrier and locks) for inter-device synchronization.
        This method ensures that these resources are initialized only once by device 0
        and then shared among all devices.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """

        # Logic: Device 0 acts as the coordinator to initialize shared synchronization primitives.
        if self.device_id == 0:

            # Invariant: `num_threads` correctly reflects the total count of active devices
            # participating in the barrier synchronization.
            num_threads = len(devices)
            # Functional Utility: `ReusableBarrierCond` acts as a synchronization point,
            # ensuring all device threads reach a specific state before proceeding.
            barrier = ReusableBarrierCond(num_threads)
            location_locks = {}

            # Block Logic: Initializes a dictionary of locks, ensuring each unique sensor
            # data location has an associated lock for concurrent access control.
            for device in devices:
                for location in device.sensor_data:
                    if location not in location_locks:
                        location_locks[location] = Lock()

            # Functional Utility: Ensures the barrier and locks are initialized once and
            # assigned to the current device if not already set.
            if self.barrier is None:
                self.barrier = barrier
                self.locks = location_locks

            # Block Logic: Propagates the initialized barrier and locks to all other devices,
            # establishing shared synchronization across the simulation.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
                if device.locks is None:
                    device.locks = location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location for this device.
        If no script is provided, it signals that the timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None if the timepoint is complete.
            location (str): The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: Signals completion of processing for the current timepoint,
            # allowing the DeviceThread to proceed with synchronization.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            any: The sensor data for the specified location, or None if not found.
        """
        data = None

        if location in self.sensor_data:
            data = self.sensor_data[location]
        return data

    def set_data(self, location, data):
        """
        Sets sensor data for a given location.

        Args:
            location (str): The location for which to set data.
            data (any): The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device thread, waiting for its completion.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the lifecycle and execution of a Device. This thread continuously
    fetches neighbor information, executes assigned scripts, and synchronizes
    with other DeviceThreads at each timepoint.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Functional Utility: `WorkerPool` enables concurrent execution of scripts
        # for different locations on a device, improving simulation throughput.
        self.pool = WorkerPool(8, self.device)

    def run(self):
        """
        The main execution loop for the DeviceThread. It continuously processes
        sensor data, executes scripts, and synchronizes with other devices
        until a shutdown signal is received.
        """
        while True:
            # Block Logic: Fetches the current set of neighboring devices from the supervisor.
            # This is crucial for collaborative script execution across devices.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Block Logic: Upon receiving a shutdown signal (neighbours is None),
                # workers are terminated, and the thread exits its loop.
                for _ in xrange(self.pool.max_threads):
                    self.pool.submit_work(None, None, None)
                self.pool.wait_completion()
                break

            # Block Logic: Halts execution until the current timepoint's scripts
            # have been assigned and marked as ready by the device.
            self.device.timepoint_done.wait()

            # Block Logic: Submits each assigned script and its associated location
            # to the worker pool for parallel execution.
            for (script, location) in self.device.scripts:
                self.pool.submit_work(neighbours, script, location)

            # Block Logic: Waits for all scripts submitted in the current timepoint
            # to complete their execution by the worker pool.
            self.pool.wait_completion()

            # Functional Utility: Synchronizes with all other DeviceThreads in the simulation
            # using a barrier, ensuring all devices complete their current timepoint
            # processing before proceeding.
            self.device.barrier.wait()

            # Block Logic: Resets the timepoint completion event, preparing for the
            # next cycle of script assignment and execution.
            self.device.timepoint_done.clear()

        # Functional Utility: Initiates the shutdown sequence for the worker pool,
        # ensuring all worker threads are gracefully terminated.
        self.pool.end_threads()


class WorkerPool(object):
    """
    Manages a pool of worker threads responsible for executing scripts
    on sensor data for a specific device. It uses a queue to distribute
    work among the threads.
    """

    def __init__(self, no_workers, device):
        """
        Initializes a new WorkerPool instance.

        Args:
            no_workers (int): The number of worker threads to create in the pool.
            device (Device): The Device instance associated with this worker pool.
        """
        self.max_threads = no_workers
        self.queue = Queue(no_workers)
        self.device = device
        self.thread_list = []
        # Block Logic: Spawns and starts a specified number of `WorkerThread` instances,
        # each configured to process tasks from the shared queue.
        for _ in range(no_workers):
            thread = WorkerThread(self.device, self.queue)
            self.thread_list.append(thread)
            thread.start()

    def submit_work(self, neighbours, script, location):
        """
        Submits a work item (script, location, and neighbor data) to the worker pool.

        Args:
            neighbours (list): A list of neighboring devices.
            script (Script): The script to be executed.
            location (str): The location pertinent to the script execution.
        """
        self.queue.put((neighbours, script, location))

    def wait_completion(self):
        """
        Waits until all submitted work items in the queue have been processed.
        """
        self.queue.join()

    def end_threads(self):
        """
        Signals all worker threads to terminate and waits for their completion.
        """
        # Block Logic: Sends termination signals to all worker threads by submitting
        # special `None` work items, allowing them to exit their run loops gracefully.
        for _ in self.thread_list:
            self.queue.put((None, None, None))
        # Block Logic: Waits for each worker thread to complete its execution and join,
        # ensuring all threads are properly shut down before proceeding.
        for thread in self.thread_list:
            thread.join()
        self.thread_list = []


class WorkerThread(Thread):
    """
    A worker thread that processes tasks from a queue. Each task involves
    executing a script at a specific location, potentially involving data
    from neighboring devices.
    """

    def __init__(self, device, tasks):
        """
        Initializes a new WorkerThread instance.

        Args:
            device (Device): The Device instance associated with this worker.
            tasks (Queue): The queue from which to retrieve tasks.
        """
        Thread.__init__(self)
        self.device = device
        self.tasks = tasks

    def run(self):
        """
        The main execution loop for the WorkerThread. It continuously retrieves
        tasks from the queue, executes the associated scripts, and updates
        sensor data.
        """
        while True:
            # Block Logic: Retrieves a task from the queue, blocking until a task is available.
            neighbours, script, location = self.tasks.get()

            # Invariant: A `None` tuple for (neighbours, script, location) serves as a sentinel value,
            # indicating that the worker thread should terminate.
            if neighbours is None and script is None and location is None:
                self.tasks.task_done()
                return

            # Functional Utility: Acquires a lock for the specific location to ensure
            # exclusive access to the sensor data during script execution, preventing
            # race conditions in a multi-threaded environment.
            with self.device.locks[location]:
                script_data = []
                # Block Logic: Gathers sensor data from neighboring devices at the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Retrieves the local device's sensor data for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: `script_data` must not be empty for script execution.
                # Post-condition: If `script_data` is not empty, the script is executed
                # and its result is propagated to neighboring and local device data.
                if script_data != []:
                    # Functional Utility: Executes the assigned script with the collected data,
                    # simulating sensor data processing.
                    result = script.run(script_data)

                    # Block Logic: Updates the sensor data of neighboring devices with the script's result.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Updates the local device's sensor data with the script's result.
                    self.device.set_data(location, result)

            # Functional Utility: Marks the current task as done in the queue,
            # signaling completion to the `WorkerPool`.
            self.tasks.task_done()
