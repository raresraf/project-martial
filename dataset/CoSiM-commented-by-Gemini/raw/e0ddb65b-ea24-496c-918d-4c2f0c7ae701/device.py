from threading import Event, Thread, Lock, Semaphore
# The 'barrier' module is imported but not provided. It is assumed to contain
# a standard reusable Barrier class for thread synchronization.
from barrier import Barrier
# The 'thread_pool' module is imported, but a ThreadPool class is also
# defined below. This implementation uses the local definition.
from thread_pool import ThreadPool


class Device(object):
    """
    Represents a device in the simulation. Each device has its own thread and
    a thread pool to execute scripts. It manages its own sensor data and a set
    of locks for each data location.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.barrier = None
        self.scripts = []
        self.timepoint_done = Event()

        # Each device has its own set of locks for its locations.
        self.locks = {}
        for location in sensor_data:
            self.locks[location] = Lock()

        self.scripts_available = False

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier for all devices.
        """
        # Device 0 is responsible for creating and distributing the barrier.
        if self.device_id == 0:
            barrier = Barrier(len(devices))
            self.barrier = barrier
            self.send_barrier(devices, barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        """A static method to assign the shared barrier to all devices."""
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """Sets the barrier for this device."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of
        the assignment phase for a timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a location after acquiring its lock.
        
        @bug This method acquires a lock but does not release it. The release
        is expected to happen in `set_data`. This pattern is unsafe and leads
        to deadlocks when one device calls `get_data` on another, as it cannot
        call the other device's `set_data` to release the lock.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Updates sensor data for a location and releases its lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating a ThreadPool to
    execute scripts for each timepoint.
    """
    NR_THREADS = 8

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(self.NR_THREADS)

    def run(self):
        # Provide the thread pool with a reference to the parent device.
        self.thread_pool.set_device(self.device)

        while True:
            # Pre-condition: Get neighbors for the new timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Block Logic: This complex loop with a flag appears to be a state
            # machine to ensure scripts are submitted only once per timepoint.
            while True:
                # Wait for the signal that all scripts for the timepoint are assigned.
                self.device.timepoint_done.wait()
                if self.device.scripts_available:
                    self.device.scripts_available = False
                    # Submit all assigned scripts to the thread pool.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task((neighbours, location, script))
                else:
                    self.device.timepoint_done.clear()
                    self.device.scripts_available = True
                    break

            # Wait for the thread pool to finish all tasks for this timepoint.
            self.thread_pool.wait()
            # Invariant: All devices synchronize here before the next timepoint.
            self.device.barrier.wait()
        
        # Cleanly shut down the thread pool.
        self.thread_pool.finish()


from threading import Thread
from Queue import Queue


class ThreadPool(object):
    """
    A thread pool that manages worker threads to execute tasks from a queue.
    """
    def __init__(self, nr_threads):
        self.device = None
        self.queue = Queue(nr_threads)
        self.thread_list = []
        self.create_threads(nr_threads)
        self.start_threads()

    def create_threads(self, nr_threads):
        """Initializes the worker threads."""
        for _ in xrange(nr_threads):
            thread = Thread(target=self.execute_task)
            self.thread_list.append(thread)

    def start_threads(self):
        """Starts all worker threads."""
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].start()

    def set_device(self, device):
        """Injects a reference to the parent device."""
        self.device = device

    def submit_task(self, task):
        """Adds a task to the execution queue."""
        self.queue.put(task)

    def execute_task(self):
        """Target function for worker threads."""
        while True:
            task = self.queue.get()
            neighbours, _, script = task

            # A sentinel task signals thread termination.
            if script is None and neighbours is None:
                self.queue.task_done()
                break

            self.run_script(task)
            self.queue.task_done()

    def run_script(self, task):
        """
        Executes a script, gathering data and disseminating results.
        @bug This function triggers a deadlock. It calls `device.get_data()` on
        neighboring devices, which acquires a lock on those devices. However,
        it has no mechanism to call `set_data()` on the neighbors to release
        those locks, causing them to be held indefinitely.
        """
        neighbours, location, script = task
        script_data = []

        # Gather data from neighbors, acquiring their locks.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location) # Acquires neighbor's lock.
                if data is not None:
                    script_data.append(data)
        
        # Gather data from self, acquiring own lock.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = script.run(script_data)
            # Update data on neighbors, releasing their locks.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)
            # Update data on self, releasing own lock.
            self.device.set_data(location, result)

    def wait(self):
        """Blocks until all tasks in the queue are processed."""
        self.queue.join()

    def finish(self):
        """Shuts down the thread pool."""
        self.wait()
        # Send termination signals to all worker threads.
        for _ in xrange(len(self.thread_list)):
            self.submit_task((None, None, None))
        # Wait for all threads to terminate.
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].join()