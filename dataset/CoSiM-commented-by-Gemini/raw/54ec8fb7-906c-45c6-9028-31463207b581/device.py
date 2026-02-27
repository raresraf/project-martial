from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads, implemented
    using semaphores and a two-phase protocol.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads reach the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a device node that uses a 'ScriptExecutorService' to run tasks.
    Synchronization is centralized through a leader device (device 0).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        
        self.barrier = None # The shared inter-device barrier.
        self.barrier_is_up = Event()
        self.location_acces = {} # Dictionary for location-based locks, stored on device 0.
        self.device0 = None # Reference to the leader device.
        self.can_receive_scripts = Lock()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources. Device 0 creates the shared barrier,
        and all other devices store a reference to device 0.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            self.barrier_is_up.set() # Signal that the barrier is ready.
        
        # All devices find and store a reference to the leader.
        for device in devices:
            if device.device_id == 0:
                self.device0 = device

    def assign_script(self, script, location):
        """Assigns a script to the device's list of tasks."""
        with self.can_receive_scripts:
            if script is not None:
                self.scripts.append((script, location))
            else:
                self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's master thread to complete."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The master thread for a Device. It creates a new ScriptExecutorService
    for each timepoint to manage script execution.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main loop: waits for scripts, uses an executor service to run them,
        and synchronizes with other devices.
        """
        timepoint = 0
        while True:
            # A new executor service is created for each timepoint.
            executor_service = ScriptExecutorService()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal.

            self.device.timepoint_done.wait() # Wait for all scripts to be assigned.
            
            # This lock prevents new scripts from being assigned while processing.
            with self.device.can_receive_scripts:
                self.device.timepoint_done.clear()

                # On first run, wait for device 0 to set up the shared barrier.
                if timepoint == 0:
                    self.device.device0.barrier_is_up.wait()
                    timepoint += 1

                # Submit all scripts for this timepoint to the executor service.
                for (script, location) in self.device.scripts:
                    executor_service.submit_job(script, self.device, location, neighbours)
                
                # Wait for all submitted jobs to complete.
                executor_service.wait_finish()

            # Synchronize with all other devices at the global barrier.
            self.device.device0.barrier.wait()

class ScriptExecutorService(object):
    """
    A service that manages the execution of scripts for a single timepoint
    by spawning worker threads and limiting concurrency with a semaphore.
    """
    def __init__(self):
        # A semaphore to limit concurrent script executions to 8.
        self.core_semaphore = Semaphore(8)
        self.executors = []

    def submit_job(self, script, device, location, neighbours):
        """
        Creates and starts a new ScriptExecutor thread for a given job.
        Blocks with a semaphore if the concurrency limit is reached.
        """
        executor = ScriptExecutor(script, device, location, neighbours, self.core_semaphore)
        self.core_semaphore.acquire() # Wait if 8 jobs are already running.
        executor.start()
        self.executors.append(executor)

    def wait_finish(self):
        """Waits for all spawned executors to complete."""
        for executor in self.executors:
            executor.join()

class ScriptExecutor(Thread):
    """A worker thread that executes a single script."""
    def __init__(self, script, device, location, neighbours, core_semaphore):
        Thread.__init__(self, name="Script Executor for device %d" % device.device_id)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script
        self.core_semaphore = core_semaphore

    def run(self):
        """The main execution logic for the worker."""
        # Lazily create location-specific locks on the leader device (device 0).
        if self.location not in self.device.device0.location_acces:
            self.device.device0.location_acces[self.location] = Lock()

        # Acquire the specific lock for this data location.
        with self.device.device0.location_acces[self.location]:
            script_data = []
            
            # Gather data from neighbors and the parent device.
            if self.neighbours:
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        script_data.append(data)
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Run script and broadcast the result.
                result = self.script.run(script_data)
                if self.neighbours:
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                self.device.set_data(self.location, result)
        
        # Release the semaphore to allow another job to start.
        self.core_semaphore.release()
