


"""
This module defines the Device, DeviceThread, and ThreadPool classes for simulating
or managing a distributed system of interconnected devices. Each Device handles
sensor data, executes scripts, and synchronizes operations with a supervisor
and neighboring devices. Concurrency is managed using threading and a thread pool.
"""

from threading import Event, Thread, Lock
from barrier import Barrier
from thread_pool import ThreadPool


class Device(object):
    """
    Represents a single device in a simulated or real-world distributed system.
    Manages its unique ID, sensor data, assigned scripts, and communication
    with a central supervisor and other devices.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        :param device_id: A unique identifier for the device.
        :param sensor_data: A dictionary containing sensor readings, keyed by location.
        :param supervisor: A reference to the central supervisor managing the devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.barrier = None
        self.scripts = []
        # Event to signal when a timepoint's scripts are ready for execution.
        self.timepoint_done = Event()

        
        # Locks to protect access to sensor data at different locations.
        self.locks = {}

        for location in sensor_data:
            self.locks[location] = Lock()

        
        # Flag to indicate if scripts are available for the current timepoint.
        self.scripts_available = False

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a human-readable string representation of the device.

        :return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up synchronization barriers for all devices.
        Only device with ID 0 initiates the barrier and distributes it.

        :param devices: A list of all Device instances in the system.
        """
        if self.device_id == 0:
            # Create a barrier for all devices to synchronize.
            barrier = Barrier(len(devices))
            self.barrier = barrier
            # Distribute the created barrier to all other devices.
            self.send_barrier(devices, barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        """
        Sends the created barrier object to all other devices.

        :param devices: A list of all Device instances.
        :param barrier: The Barrier object to distribute.
        """
        # Iterate through all devices and assign the barrier, excluding the initiator.
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """
        Sets the synchronization barrier for this device.

        :param barrier: The Barrier object for synchronization.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific sensor data location.
        If a script is assigned, it enables script processing; otherwise, it
        signals that no scripts are pending for the current timepoint.

        :param script: The script (callable) to execute.
        :param location: The sensor data location associated with the script.
        """
        if script is not None:
            # Add the script and its location to the list of pending scripts.
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            # If no script, signal that this timepoint is done for this device.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Safely retrieves sensor data for a specified location using a reentrant lock.

        :param location: The key identifying the sensor data location.
        :return: The sensor data value, or None if the location is not found.
        """
        if location in self.sensor_data:
            # Acquire the lock to ensure exclusive access to the sensor data.
            self.locks[location].acquire()
            return self.sensor_data[location]

        return None

    def set_data(self, location, data):
        """
        Safely updates sensor data for a specified location and releases the lock.

        :param location: The key identifying the sensor data location.
        :param data: The new data value to set.
        """
        if location in self.sensor_data:
            # Update the sensor data.
            self.sensor_data[location] = data
            # Release the lock, allowing other threads to access the data.
            self.locks[location].release()

    def shutdown(self):
        """
        Initiates a graceful shutdown of the device's operational thread.
        """
        # Wait for the device's thread to complete its execution.
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the execution lifecycle of a Device, including fetching neighbor
    information, processing assigned scripts using a thread pool, and
    synchronizing with other devices at various timepoints.
    """
    NR_THREADS = 8

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        :param device: The Device instance this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        self.thread_pool = ThreadPool(self.NR_THREADS)

    def run(self):
        """
        The main execution loop for the device thread.
        It continuously fetches neighbor data, processes scripts in timepoints,
        and synchronizes with other devices using a barrier.
        """
        self.thread_pool.set_device(self.device)

        # Main loop for continuous operation, processing timepoints.
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Break loop if supervisor signals termination.
                break

            # Inner loop to process all scripts for the current timepoint.
            while True:
                # Wait for scripts to be assigned or for a timepoint to be marked as done.
                self.device.timepoint_done.wait()

                if self.device.scripts_available:
                    # Reset the flag and process each assigned script.
                    self.device.scripts_available = False

                    # Iterate through assigned scripts and submit them to the thread pool.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task((neighbours, location, script))
                    self.device.scripts = [] # Clear scripts after submission
                else:
                    # Reset timepoint event and availability flag for the next cycle.
                    self.device.timepoint_done.clear()
                    self.device.scripts_available = True
                    # Break from inner loop to wait for next timepoint.
                    break

            
            # Wait for all tasks in the thread pool to complete before proceeding.
            self.thread_pool.wait()

            
            # Synchronize with other devices using the global barrier.
            # Invariant: All devices reach this point before proceeding to the next timepoint.
            self.device.barrier.wait()

        
        # Signal the thread pool to finish its operations and shut down its worker threads.
        self.thread_pool.finish()


# The following classes (ThreadPool, Barrier - though Barrier is imported) would typically be in separate files
# but are included here for context based on the provided snippet.

# Assuming Barrier class definition exists in 'barrier.py'
# from barrier import Barrier 

# Assuming ThreadPool class definition was also provided as part of the context
# (It was, but I'm placing it here for logical separation as per normal Python practice)
# from thread_pool import ThreadPool

# Note: The original snippet included ThreadPool definition inline, I'm keeping it for the replacement.
from threading import Thread
from Queue import Queue


class ThreadPool(object):
    """
    Manages a pool of worker threads to execute tasks concurrently.
    Tasks are submitted to a queue and processed by available threads.
    """
    def __init__(self, nr_threads):
        """
        Initializes the ThreadPool with a specified number of worker threads.

        :param nr_threads: The number of threads in the pool.
        """
        self.device = None # The Device instance associated with this thread pool.

        self.queue = Queue(nr_threads) # Task queue with a size limit.
        self.thread_list = []

        # Create and start the worker threads.
        self.create_threads(nr_threads)
        self.start_threads()

    def create_threads(self, nr_threads):
        """
        Creates the specified number of worker threads, setting their target
        function to `execute_task`.

        :param nr_threads: The number of threads to create.
        """
        for _ in xrange(nr_threads):
            thread = Thread(target=self.execute_task)
            self.thread_list.append(thread)

    def start_threads(self):
        """
        Starts all the worker threads in the pool.
        """
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].start()

    def set_device(self, device):
        """
        Associates a Device instance with this thread pool, allowing worker
        threads to access device-specific data.

        :param device: The Device instance.
        """
        self.device = device

    def submit_task(self, task):
        """
        Adds a new task to the queue for execution by an available worker thread.

        :param task: A tuple containing (neighbours, location, script) for the task.
        """
        self.queue.put(task)

    def execute_task(self):
        """
        The main loop for each worker thread. It continuously fetches tasks
        from the queue, executes them, and signals task completion.
        """
        while True:
            # Retrieve a task from the queue. This call blocks if the queue is empty.
            task = self.queue.get()

            neighbours = task[0]
            script = task[2]

            # Check for a shutdown signal (None, None, None) task.
            if script is None and neighbours is None:
                self.queue.task_done()
                break

            # Execute the script associated with the task.
            self.run_script(task)
            # Signal that the current task has been completed.
            self.queue.task_done()

    def run_script(self, task):
        """
        Executes a script using data gathered from the local device and its neighbors.
        Updates sensor data on both the local and neighboring devices with the result.

        :param task: A tuple (neighbours, location, script) containing task details.
        """
        neighbours, location, script = task
        script_data = []

        # Gather sensor data from neighboring devices for the script execution.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Gather local device's sensor data for the script execution.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Execute the script with the gathered data.
            # Invariant: 'script' is a callable object with a 'run' method.
            result = script.run(script_data)

            # Update sensor data on neighboring devices with the script's result.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

            # Update local device's sensor data with the script's result.
            self.device.set_data(location, result)

    def wait(self):
        """
        Blocks until all tasks currently in the queue have been processed by the worker threads.
        """
        self.queue.join()

    def finish(self):
        """
        Shuts down the thread pool gracefully. It signals all worker threads
        to terminate and waits for them to complete.
        """
        # Ensure all pending tasks are processed before shutting down.
        self.wait()

        # Submit a "poison pill" task for each thread to signal termination.
        for _ in xrange(len(self.thread_list)):
            self.submit_task((None, None, None))

        # Wait for all worker threads to gracefully terminate.
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].join()

