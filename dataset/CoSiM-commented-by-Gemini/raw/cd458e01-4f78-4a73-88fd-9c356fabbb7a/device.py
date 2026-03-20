"""
This module simulates a distributed network of devices that process sensor data.

The simulation is built on a multi-threaded architecture where each device runs
in its own thread. It features several concurrency primitives to manage the
distributed state:
- A ThreadPool for each device to execute data processing scripts concurrently.
- A system-wide set of Semaphores, one for each data "location", to ensure
  that only one script is processing a given location at a time, preventing
  race conditions.
- A reusable Barrier that synchronizes all devices at the end of each discrete
  time step, ensuring that all computations for a step are complete before
  the next step begins.

The core components are:
- Device: Represents a node in the network with its own data and scripts.
- DeviceThread: The main control loop for a device.
- ThreadPool: Manages a pool of worker threads for script execution.
- Sem: A manager for location-specific semaphores.
- ReusableBarrierCond: A custom barrier implementation for synchronization.
"""


from threading import Thread, Semaphore, Condition
# Note: The following import is redundant as ThreadPool is defined in this file.
# This may be an artifact from previous code structure.
from pool_of_threads import ThreadPool

class Sem(object):
    """
    A manager for location-based semaphores.

    This class creates a dictionary mapping each unique data 'location' across
    all devices to a unique Semaphore. This ensures that operations on a specific
    location are mutually exclusive across the entire system.
    """

    def __init__(self, devices):
        """
        Initializes semaphores for all unique locations found in the devices.
        :param devices: A list of all Device objects in the simulation.
        """
        self.location_semaphore = {}
        for device in devices:
            for location in device.sensor_data:
                if location not in self.location_semaphore:
                    # Each semaphore allows only one thread access at a time.
                    self.location_semaphore[location] = Semaphore(value=1)

    def acquire(self, location):
        """Acquires the semaphore for a given location, blocking if unavailable."""
        self.location_semaphore[location].acquire()

    def release(self, location):
        """Releases the semaphore for a given location."""
        self.location_semaphore[location].release()


class ReusableBarrierCond(object):
    """
    A reusable barrier for synchronizing a fixed number of threads.

    All threads calling wait() will block until the specified number of threads
    have all called wait(). The barrier then resets for the next use.
    This is implemented using a Condition variable.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()


    def wait(self):
        """
        Blocks the calling thread until all threads have reached the barrier.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread has arrived; notify all waiting threads.
            self.cond.notify_all()
            # Reset the barrier for the next use.
            self.count_threads = self.num_threads
        else:
            # Not all threads have arrived yet; wait to be notified.
            self.cond.wait()
        self.cond.release()

class Device(object):
    """Represents a single device in the simulated network."""

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.
        :param device_id: Unique identifier for the device.
        :param sensor_data: A dictionary of {location: data} for this device.
        :param supervisor: An object responsible for providing network topology (neighbours).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List of (script, location) tuples to execute.

        # Concurrency primitives shared across devices.
        self.barrier = None
        self.location_semaphore = None
        self.timepoint_done = False # Flag to signal start of a time step.

        # Each device has its own master thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared concurrency objects for all devices in the simulation.

        This method should be called on one device (e.g., device 0) to initialize
        and distribute the shared barrier and semaphores to all other devices.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            self.location_semaphore = Sem(devices)

            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier
                    device.location_semaphore = self.location_semaphore

    def assign_script(self, script, location):
        """
        Assigns a script to be run by this device in the next time step.
        A `None` script is a signal that all scripts for the timepoint are assigned.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done = True

    def get_data(self, location):
        """Retrieves sensor data for a given location from this device."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location on this device."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the device's master thread to gracefully shut it down."""
        self.thread.join()

class DeviceThread(Thread):
    """The main control thread for a single Device."""

    def __init__(self, device):
        """
        Initializes the device thread.
        :param device: The Device object this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Each device has its own thread pool for executing scripts.
        self.thread_pool = ThreadPool(8, device)

    def run(self):
        """
        The main simulation loop for the device.
        This loop represents the progression of discrete time steps.
        """
        while True:
            # Get the current network topology for this device.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals end of simulation.
                break

            # Wait until all scripts for the current timepoint are assigned.
            while True:
                if self.device.timepoint_done:
                    # Reset flag and submit all assigned scripts to the thread pool.
                    self.device.timepoint_done = False
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
                    break

            # Wait for this device's thread pool to finish all submitted tasks.
            self.thread_pool.wait_threads()

            # Wait at the barrier for all other devices to finish their time step.
            self.device.barrier.wait()

        # Cleanly shut down the thread pool and its worker threads.
        self.thread_pool.end_threads()


from Queue import Queue
from threading import Thread

class ThreadPool(object):
    """
    A simple thread pool implementation to execute data processing jobs.
    """
    def __init__(self, threads_count, device):
        """
        Initializes the thread pool and starts the worker threads.
        :param threads_count: The number of worker threads to create.
        :param device: The parent device, needed to access shared resources.
        """
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = device
        self.create_and_start_worker_threads(threads_count)

    def create_and_start_worker_threads(self, threads_count):
        """Creates and starts the pool of worker threads."""
        for _ in range(threads_count):
            thread = Thread(target=self.do_job)
            self.threads.append(thread)

        for thread in self.threads:
            thread.start()

    def do_job(self):
        """
        The target function for worker threads.
        Continuously pulls jobs from the queue and executes them.
        """
        while True:
            # Get a job from the queue. This blocks until a job is available.
            neighbours, script, location = self.queue.get()

            # Sentinel check: A None job signals the thread to terminate.
            if neighbours is None and script is None and location is None:
                self.queue.task_done()
                return

            # --- Critical Section for a Location ---
            # Acquire the semaphore for the job's location to ensure exclusive access.
            self.device.location_semaphore.acquire(location)
            
            # Gather data from this device and all its neighbours for the given location.
            script_data = []
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data: # Only run script if there is data to process.
                # Execute the script on the collected data.
                result = script.run(script_data)

                # Broadcast the result by updating the data on all involved devices.
                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            
            # Release the semaphore so other jobs for this location can run.
            self.device.location_semaphore.release(location)
            # --- End of Critical Section ---

            self.queue.task_done()

    def submit(self, neighbours, script, location):
        """Adds a new job to the queue for the thread pool to execute."""
        self.queue.put((neighbours, script, location))

    def wait_threads(self):
        """Blocks until all jobs in the queue have been completed."""
        self.queue.join()

    def end_threads(self):
        """Shuts down the thread pool by sending a sentinel job to each thread."""
        for _ in range(len(self.threads)):
            self.submit(None, None, None)

        self.wait_threads() # Wait for sentinel jobs to be processed.

        for thread in self.threads:
            thread.join()
