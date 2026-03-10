
"""
This module provides a simulation framework for distributed devices,
utilizing multi-threading and synchronization primitives. It includes
a reusable barrier, a device abstraction managing sensor data and scripts,
and a worker pool for concurrent script execution.
"""

from threading import Lock, Thread, Semaphore, Event
from Queue import Queue # Note: In Python 3, this would be 'queue'


class ReusableBarrier(object):
    """
    A reusable barrier synchronization primitive for a fixed number of threads.

    This barrier allows a set of threads to wait for each other to reach a common
    point before any of them can proceed. It is designed to be reusable across
    multiple synchronization points within a larger simulation loop.

    Algorithm: Implements a double-barrier pattern using two semaphores and a lock.
               The use of single-element lists (`count_threads1`, `count_threads2`)
               as mutable counters is a Python-specific idiom to allow shared integer
               state modification across method calls within the `count_lock` context.
    """
    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must arrive
                               at the barrier before any can proceed.
        """
        self.num_threads = num_threads
        # Mutable integer counters wrapped in lists to allow modification by reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # Lock to protect the counter during updates to ensure atomicity.
        self.count_lock = Lock()
        # Semaphores for the two phases of the barrier. Initialized to 0 so threads block until released.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        `num_threads` threads have also called `wait()`.
        """
        
        # Executes the first phase of the barrier.
        self.phase(self.count_threads1, self.threads_sem1)
        # Executes the second phase of the barrier, ensuring reusability.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Manages one phase of the double-barrier synchronization.

        Block Logic: Decrements a shared counter protected by a lock.
                     When the counter reaches zero, it indicates all `num_threads`
                     have arrived, and then all threads are released via the semaphore.
                     The counter is then reset for the next use.

        Args:
            count_threads (list): A single-element list holding the mutable counter for this phase.
            threads_sem (Semaphore): The semaphore used to block and release threads for this phase.
        """
        
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # If all threads have arrived, release the semaphore for each thread.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # Acquire the semaphore; this will block the current thread until it's released by the last arriving thread.
        threads_sem.acquire()


        threads_sem.acquire()


class Device(object):
    """
    Represents a simulated device within a distributed system.

    Each device manages its sensor data, accepts scripts for execution,
    and coordinates with a central supervisor. It utilizes a dedicated
    thread and a worker pool to concurrently process tasks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor readings
                                (e.g., {location_id: data_value}).
            supervisor (object): An object representing the central supervisor,
                                 used for coordination (e.g., getting neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data

        self.supervisor = supervisor
        # Event to signal that a script has been received and processed.
        self.script_received = Event()
        # List to store assigned scripts for the current timepoint.
        self.scripts = []
        # Reference to the global barrier used for synchronizing all devices.
        self.barrier = None

        # Dictionary of locks, one for each sensor data location, to protect
        # data during concurrent access from multiple threads/devices.
        self.locks = {}
        for spot in sensor_data:
            self.locks[spot] = Lock()
        # Event to signal when all scripts for the current timepoint have been assigned.
        self.timepoint_done = Event()
        # The main thread for the device, responsible for supervisor interaction and job dispatch.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global barrier for all devices in the simulation.

        This method is called once during initialization. Only device 0
        is responsible for creating and distributing the barrier instance.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Only device with ID 0 initializes the global barrier.
        # This acts as the coordinator for the global synchronization.
        if self.device_id == 0:
            # Initializes the ReusableBarrier with the total number of devices.
            self.barrier = ReusableBarrier(len(devices))
            # Distributes the same barrier instance to all other devices.
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific data location.

        Scripts are stored internally and later dispatched to worker threads.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location the script operates on.
        """
        if script is not None:
            # Appends the script and its location to a list for later processing.
            self.scripts.append((script, location))
        else:
            # If script is None, it signifies the end of script assignments for the current timepoint.
            # Functional Utility: Signals that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location, acquiring a lock for that location.

        Note: This implementation implies that the caller is responsible for
        releasing the lock via a subsequent `set_data` call for the same location,
        or explicitly by directly calling `self.locks[location].release()`.
        This pattern can be prone to deadlocks if not handled carefully.

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if not found.
        """
        for loc in self.sensor_data:
            if loc == location:
                # Functional Utility: Acquires a lock for the specific data location,
                # ensuring exclusive access to the data.
                self.locks[loc].acquire()
                return self.sensor_data[loc]

        return None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location, and releases the lock.

        Pre-condition: A lock for 'location' must have been previously acquired
                       by a call to `get_data(location)`.

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Functional Utility: Releases the lock for the specific data location,
            # allowing other threads/devices to access it.
            self.locks[location].release()

    def shutdown(self):
        """
        Performs a graceful shutdown of the main device thread.
        """
        # Waits for the main device thread to complete its execution.
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for coordinating with the supervisor,
    dispatching scripts to a worker pool, and participating in global synchronization.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Functional Utility: Creates a pool of worker threads for this device to execute scripts concurrently.
        self.dev_threads = ThreadsForEachDevice(8) # Invariant: A fixed number of 8 worker threads per device.

    def run(self):
        """
        Main execution loop for the DeviceThread.

        Architectural Intent: Continuously fetches neighbor information from the supervisor,
        distributes assigned scripts to its worker thread pool via a queue, and
        participates in global barrier synchronization for timepoint progression.
        """
        # Provides the worker pool with a reference to the parent Device.
        self.dev_threads.device = self.device

        while True:
            # Block Logic: Fetches the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Termination Condition: If no neighbors are returned (None), it signifies
            # that the simulation is ending for this device, and the loop breaks.
            if neighbours is None:
                break

            # Block Logic: Waits until all scripts for the current timepoint have been assigned to the device.
            self.device.timepoint_done.wait()

            # Block Logic: Iterates through all assigned scripts and enqueues them for execution by worker threads.
            for (script, location) in self.device.scripts:
                self.dev_threads.jobs_to_be_done.put(
                    (neighbours, script, location))

            # Functional Utility: Resets the list of scripts and the timepoint_done event for the next cycle.
            self.device.scripts = []
            self.device.timepoint_done.clear()

            # Functional Utility: Waits until all jobs (scripts) dispatched in this timepoint
            # have been processed by the worker threads.
            self.dev_threads.jobs_to_be_done.join()
            
            # Block Logic: Waits at the global barrier for all devices to complete their
            # current timepoint processing before proceeding.
            self.device.barrier.wait()

        # Functional Utility: After the main loop breaks (supervisor returned None),
        # waits for any remaining jobs in the queue to complete.
        self.dev_threads.jobs_to_be_done.join()

        # Block Logic: Sends termination signals (None, None, None) to all worker threads
        # to gracefully shut them down.
        for _ in range(len(self.dev_threads.threads)):
            self.dev_threads.jobs_to_be_done.put((None, None, None))

        # Block Logic: Waits for all worker threads to terminate.
        for d_th in self.dev_threads.threads:
            d_th.join()


class ThreadsForEachDevice(object):
    """
    Manages a pool of worker threads for a single Device.

    These threads are responsible for dequeuing tasks (scripts with associated data locations),
    collecting relevant sensor data from the device itself and its neighbors,
    executing the scripts, and updating sensor data.
    """

    def __init__(self, number_of_threads):
        """
        Initializes the thread pool.

        Args:
            number_of_threads (int): The number of worker threads to create in the pool.
        """

        self.device = None # Will be set by the DeviceThread after initialization.
        
        # Queue to hold jobs (scripts, locations, and neighbor info) to be processed by worker threads.
        # The queue size is typically set to the number of threads for optimal performance, preventing
        # excessive memory usage for queued tasks.
        self.jobs_to_be_done = Queue(number_of_threads)
        self.threads = [] # List to hold references to the worker thread objects.

        # Functional Utility: Sets up the worker threads and starts them.
        self.create_threads(number_of_threads)
        self.start_threads()

    def create_threads(self, number_of_threads):
        """
        Creates the specified number of worker threads.
        Each thread is configured to run the `execute` method.

        Args:
            number_of_threads (int): The total number of worker threads to instantiate.
        """
        for _ in range(number_of_threads):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

    def start_threads(self):
        """
        Starts all worker threads in the pool.
        """
        for i_th in self.threads:
            i_th.start()

    def execute(self):
        """
        The main loop for each worker thread.

        Architectural Intent: Continuously retrieves jobs from the queue,
        collects necessary data, executes the script, and updates results
        on the device and its neighbors.
        """
        while True:
            # Block Logic: Retrieves a job (neighbours, script, location) from the queue.
            # This call blocks until a job is available.
            neighbours, script, location = self.jobs_to_be_done.get()
            
            # Termination Condition: If a (None, None, None) tuple is received,
            # it signals the thread to terminate gracefully.
            if neighbours is None and script is None:
                self.jobs_to_be_done.task_done() # Signals that this job is complete (for queue's join method).
                return

            data_for_script = []
            
            # Block Logic: Collects sensor data from neighboring devices at the specified location.
            for device in neighbours:
                # Ensures data is not collected from itself if present in neighbors list.
                if device.device_id != self.device.device_id:
                    data = device.get_data(location) # Note: get_data acquires a lock.
                    if data is not None:
                        data_for_script.append(data)
            
            # Block Logic: Collects sensor data from its own device at the specified location.
            data = self.device.get_data(location) # Note: get_data acquires a lock.
            if data is not None:
                data_for_script.append(data)

            # Block Logic: If data was collected, executes the script and updates data.
            if data_for_script != []:
                
                scripted_data = script.run(data_for_script) # Functional Utility: Executes the assigned script with collected data.

                
                # Updates data on neighboring devices.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, scripted_data) # Note: set_data releases a lock.

                
                # Updates data on its own device.
                self.device.set_data(location, scripted_data) # Note: set_data releases a lock.

            # Functional Utility: Signals to the queue that the current job is finished,
            # allowing the DeviceThread to eventually proceed after all jobs are done.
            self.jobs_to_be_done.task_done()
