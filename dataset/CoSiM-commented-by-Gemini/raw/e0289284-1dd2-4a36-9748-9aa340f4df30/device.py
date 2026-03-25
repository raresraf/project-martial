from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

class ThreadPool(object):
    """
    A simple thread pool that manages a fixed number of worker threads to
    execute tasks from a queue.
    """
    def __init__(self, thread_number, device):
        """
        Initializes and starts the worker threads.

        Args:
            thread_number (int): The number of worker threads to create.
            device (Device): A reference to the parent device object, used by
                             tasks executed in the pool.
        """
        self.thread_number = thread_number
        self.queue = Queue(self.thread_number)
        self.threads = []
        self.device = device

        # Create and start the worker threads.
        for _ in xrange(thread_number):
            self.threads.append(Thread(target=self.execute))
        for thread in self.threads:
            thread.start()

    def execute(self):
        """
        The target function for worker threads. It continuously fetches tasks
        from the queue and executes them until a sentinel value (None) is received.
        """
        # A sentinel value of (None, None, None) signals termination.
        neighbours, script, location = self.queue.get()
        while neighbours is not None and script is not None and location is not None:
            # Run the core task logic.
            self.run(neighbours, script, location)
            # Signal that the task from the queue is done.
            self.queue.task_done()
            # Get the next task.
            neighbours, script, location = self.queue.get()
        # Signal that the final sentinel task is done.
        self.queue.task_done()

    def run(self, neighbours, script, location):
        """
        Executes a single script. It acquires a lock for the script's location,
        gathers data, runs the script, and disseminates the results.

        Args:
            neighbours (list): A list of neighboring Device objects.
            script (object): The script to be executed.
            location (any): The location context for the script.
        """
        script_data = []
        # Functional Utility: A location-specific lock ensures that multiple
        # scripts targeting the same data location do not cause race conditions.
        self.device.location_lock[location].acquire()
        
        # Gather data from all neighbors.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Gather data from the parent device itself.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = script.run(script_data)
            # Disseminate the result to all participating devices.
            for device in neighbours:
                device.set_data(location, result)
            self.device.set_data(location, result)
        
        self.device.location_lock[location].release()

    def submit(self, neighbours, script, location):
        """Submits a new task to the thread pool's queue."""
        self.queue.put((neighbours, script, location))

    def wait(self):
        """Blocks until all tasks in the queue are processed."""
        self.queue.join()

    def end(self):
        """Shuts down the thread pool gracefully."""
        # Wait for any remaining tasks to complete.
        self.wait()
        # Send a sentinel value for each thread to signal termination.
        for _ in xrange(self.thread_number):
            self.submit(None, None, None)
        # Wait for all worker threads to finish.
        for thread in self.threads:
            thread.join()


class Barrier(object):
    """
    A standard, reusable two-phase barrier for thread synchronization.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all participating threads have called wait."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """Represents a single device in the simulation."""
    def __init__(self, device_id, sensor_data, supervisor):
        # A list to hold shared locks for specific data locations.
        self.location_lock = [None] * 100
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.all_devices = None
        self.recived_flag = False

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes shared resources across all devices."""
        self.all_devices = devices
        # The first device to call setup creates and distributes the global barrier.
        if self.barrier is None:
            self.barrier = Barrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device and handles the creation of shared locks.
        """
        if script is not None:
            # Block Logic: This complex mechanism ensures that all devices share the
            # exact same Lock object for any given location.
            # @note This is an inefficient O(N) operation for every new location.
            if self.location_lock[location] is None:
                self.location_lock[location] = Lock()
                self.recived_flag = True
                # Propagate the newly created lock to all other devices.
                for device_number in xrange(len(self.all_devices)):
                    self.all_devices[device_number].location_lock[location] = self.location_lock[location]
            
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of assignments for the current timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, which orchestrates a pool of
    worker threads to perform the actual script executions.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Each device has its own thread pool for concurrent script execution.
        self.thread_pool = ThreadPool(8, self.device)

    def run(self):
        """
        The main simulation loop. It synchronizes timepoints, manages script
        submission to the thread pool, and waits for completion.
        """
        while True:
            # Get neighbors for the new timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Block Logic: This inner loop with a flag appears to be a complex
            # way to ensure scripts are submitted to the pool only once per
            # timepoint signal.
            while True:
                # Wait for the signal that script assignment is done.
                self.device.timepoint_done.wait()
                
                if self.device.recived_flag:
                    # Submit all assigned scripts to the thread pool for execution.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
                    self.device.recived_flag = False
                else:
                    self.device.timepoint_done.clear()
                    self.device.recived_flag = True
                    break

            # Wait for all submitted tasks in the current timepoint to finish.
            self.thread_pool.wait()

            # Invariant: All devices synchronize here before proceeding to the next timepoint.
            self.device.barrier.wait()
        
        # Cleanly shut down the thread pool at the end of the simulation.
        self.thread_pool.end()