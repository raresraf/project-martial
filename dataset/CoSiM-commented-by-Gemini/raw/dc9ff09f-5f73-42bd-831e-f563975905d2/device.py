from threading import Event, Thread, Lock, Semaphore
from multiprocessing import cpu_count
from Queue import Queue


class ReusableBarrierSem(object):
    """
    A reusable barrier implemented using semaphores, for synchronizing a fixed
    number of threads. It employs a two-phase signaling mechanism to ensure
    that threads from a previous wait do not interfere with threads in a
    subsequent wait.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a specified number of threads.
        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        # Counters for threads entering each phase.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads

        # Lock to protect access to the counters.
        self.counter_lock = Lock()

        # Semaphores to block threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks the calling thread until all threads have reached this point.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread to arrive, release all waiting threads.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset for the next use.
                self.count_threads1 = self.num_threads
        # Wait for the last thread to release the semaphore.
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread to arrive, release all waiting threads.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset for the next use.
                self.count_threads2 = self.num_threads
        # Wait for the last thread to release the semaphore.
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a device in a simulated environment, which uses a pool of
    threads to process scripts concurrently.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.
        Args:
            device_id (int): Unique identifier for the device.
            sensor_data (dict): Initial sensor data for the device.
            supervisor (object): The supervisor managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # List of scripts to be executed in the current timepoint.
        self.scripts = []

        # Event to signal the completion of a timepoint's script assignments.
        self.timepoint_done = Event()

        # List of neighboring devices.
        self.neighbours = []

        # Determine the number of threads for the device's thread pool.
        self.num_of_threads = cpu_count()
        if self.num_of_threads < 8:
            self.num_of_threads = 8

        # Queue for tasks (scripts) to be executed by the thread pool.
        self.tasks = Queue()

        # Semaphore to control access to the task queue.
        self.semaphore = Semaphore(0)

        # Semaphores for locking individual locations.
        self.num_locations = self.supervisor.supervisor.testcase.num_locations
        self.lock_locations = []

        # Lock for ensuring thread-safe access to the task queue.
        self.lock_queue = Lock()

        # Barrier for synchronizing threads within this device's pool.
        self.barrier = ReusableBarrierSem(self.num_of_threads)

        # Barrier for synchronizing all devices across the simulation.
        self.global_barrier = ReusableBarrierSem(0)

        # The device's internal thread pool.
        self.pool = Pool(self)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for all devices in the simulation. This method
        should be called by one device (e.g., device 0) to initialize shared
        state.
        Args:
            devices (list): A list of all device objects in the simulation.
        """
        if self.device_id == 0:
            # Device 0 is responsible for initializing the global barrier
            # and location locks, which are then shared with all other devices.
            self.global_barrier = ReusableBarrierSem(len(devices))

            for _ in range(self.num_locations):
                self.lock_locations.append(Semaphore(1))

            for device in devices:
                device.global_barrier = self.global_barrier
                device.lock_locations = self.lock_locations

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.
        Args:
            script (object): The script to execute.
            location (int): The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for the timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.
        Args:
            location (int): The location to query.
        Returns:
            The data at the specified location, or None if not available.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates sensor data at a given location.
        Args:
            location (int): The location to update.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread pool."""
        self.pool.shutdown()


class Pool(object):
    """A thread pool for managing worker threads (DeviceThread)."""

    def __init__(self, device):
        """
        Initializes the thread pool.
        Args:
            device (Device): The device this pool belongs to.
        """
        self.device = device
        self.thread_list = []

        # Create and start the worker threads.
        for i in range(self.device.num_of_threads):
            self.thread_list.append(DeviceThread(self.device, i))
        for thread in self.thread_list:
            thread.start()

    def add_task(self, task):
        """
        Adds a task to the queue for the worker threads.
        Args:
            task (any): The task to be added.
        """
        self.device.tasks.put(task)
        self.device.semaphore.release()

    def shutdown(self):
        """Waits for all worker threads to complete."""
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread that executes scripts for a device.
    """

    def __init__(self, device, thread_id):
        """
        Initializes a device thread.
        Args:
            device (Device): The device this thread belongs to.
            thread_id (int): The ID of this thread within the pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id
        self.script = None
        self.location = None
        self.script_data = []
        self.data = None
        self.result = None

    def run(self):
        """
        The main loop for the worker thread. It waits for tasks, executes them,
        and synchronizes with other threads and devices.
        """
        while True:
            # Block Logic: In each timepoint, thread 0 is responsible for setup.
            if self.thread_id == 0:
                self.device.timepoint_done.clear()
                # Pre-condition: Fetches neighbors for the upcoming timepoint.
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # Invariant: All threads in the pool synchronize here before proceeding.
            self.device.barrier.wait()

            # A None neighbours list is the signal to terminate.
            if self.device.neighbours is None:
                break

            # Block Logic: Thread 0 acts as the leader for the timepoint.
            if self.thread_id == 0:
                # It waits until all scripts for the timepoint are assigned.
                self.device.timepoint_done.wait()
                # It then adds all scripts as tasks to the queue.
                for (script, location) in self.device.scripts:
                    self.device.pool.add_task((script, location))

                # Release the semaphore for each worker thread to indicate that
                # tasks are available.
                for _ in range(self.device.num_of_threads):
                    self.device.semaphore.release()

            # Block Logic: Worker threads process tasks from the queue.
            while True:
                # Wait for a task to be available.
                self.device.semaphore.acquire()

                # Atomically get a task from the queue.
                with self.device.lock_queue:
                    if not self.device.tasks.empty():
                        (self.script, self.location) = self.device.tasks.get()
                    else:
                        # If the queue is empty, exit the loop for this timepoint.
                        break

                # Acquire a lock for the specific location to ensure data consistency.
                self.device.lock_locations[self.location].acquire()
                self.script_data = []

                # Gather data from neighbors at the specified location.
                for device in self.device.neighbours:
                    self.data = device.get_data(self.location)
                    if self.data is not None:
                        self.script_data.append(self.data)

                # Include self in the data gathering.
                self.data = self.device.get_data(self.location)
                if self.data is not None:
                    self.script_data.append(self.data)

                # Execute the script and update data on all relevant devices.
                if self.script_data:
                    self.result = self.script.run(self.script_data)
                    for device in self.device.neighbours:
                        device.set_data(self.location, self.result)
                    self.device.set_data(self.location, self.result)

                # Release the location lock.
                self.device.lock_locations[self.location].release()

            # Block Logic: Thread 0 synchronizes all devices globally.
            if self.thread_id == 0:
                self.device.global_barrier.wait()

            # Invariant: All threads within the pool synchronize before the next timepoint.
            self.device.barrier.wait()