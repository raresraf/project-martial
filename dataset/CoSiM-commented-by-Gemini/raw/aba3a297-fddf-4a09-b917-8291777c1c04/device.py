"""
A simulation framework for a network of communicating devices using a
thread pool for concurrent script execution.

This script defines a system where each `Device` uses a dedicated `ThreadPool`
to process assigned scripts. This approach allows for a fixed number of worker
threads to handle an arbitrary number of tasks, which is generally more
efficient than creating a new thread for every task.

The main components are:
- Device: Represents a node in the network. It manages its own `ThreadPool`
  and a set of per-location locks for its sensor data.
- DeviceThread: The main control loop for a `Device`. It orchestrates the
  time-steps, waits for script assignments, submits them to the thread pool,
  and handles global synchronization with other devices.
- ThreadPool: A classic thread pool implementation using a `Queue` to manage
  tasks. Worker threads fetch tasks, execute them, and signal completion.

Key Architectural Points:
- Each device has its own thread pool, allowing for intra-device parallelism.
- Synchronization between devices is handled by a shared `ReusableBarrierCond`.
- Data access is controlled by per-location locks. This is a granular
  locking strategy but introduces a significant risk of deadlock, as worker
  threads acquire locks on multiple devices during data gathering.
"""
from threading import Event, Thread, Lock
from Queue import Queue

# These imports are assumed to exist in the execution environment.
from barrier import ReusableBarrierCond
from threadpool import ThreadPool as ExternalThreadPool # Renamed to avoid conflict

class Device(object):
    """
    Represents a device node in the simulation.

    Each device has its own thread, a pool of worker threads for script
    execution, and a set of locks to manage concurrent access to its
    sensor data locations.

    Attributes:
        device_id (int): Unique identifier for the device.
        sensor_data (dict): Local data store.
        supervisor (object): The central simulation controller.
        scripts (list): A list of (script, location) tuples for the current step.
        locations_locks (dict): A dictionary mapping data locations to Locks.
        barrier (ReusableBarrierCond): A shared barrier for inter-device sync.
        thread (DeviceThread): The main control thread for this device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        # Create a dictionary of locks, one for each sensor data location.
        self.locations_locks = {
            location: Lock() for location in sensor_data
        }

        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared barrier for all devices.

        This method should be called on one primary device (e.g., device_id 0),
        which then creates and distributes the barrier to all other devices.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to the device for the current time-step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that all scripts for the step are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Safely retrieves sensor data by acquiring the location-specific lock.

        NOTE: This method blocks until the lock is acquired. The lock is released
        in `set_data`. This protocol creates a high risk of deadlock if multiple
        threads attempt to acquire locks on multiple devices in different orders.
        """
        if location in self.sensor_data:
            self.locations_locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Safely sets sensor data and releases the location-specific lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations_locks[location].release()

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device, orchestrating script execution
    via a thread pool.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Each device gets its own thread pool.
        self.thread_pool = ThreadPool(8, device)

    def run(self):
        """
        The main simulation loop for the device.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals termination.
                break

            # This complex loop waits for the supervisor to signal that all
            # scripts have been assigned for the current time-step.
            while True:
                # Wait until the timepoint_done event is set.
                if self.device.timepoint_done.wait() and not self.device.script_received.is_set():
                    self.device.timepoint_done.clear()
                    self.device.script_received.set()
                    break

                if self.device.script_received.is_set():
                    self.device.script_received.clear()
                    # Submit all collected scripts to the thread pool for execution.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task(script, location, neighbours)

            # Wait for all tasks in the queue to be processed by the workers.
            self.thread_pool.tasks_queue.join()

            # --- Global Sync Point ---
            # Wait for all other devices to finish their time-step.
            self.device.barrier.wait()

        # Cleanly shut down the thread pool once the simulation is over.
        self.thread_pool.join_threads()


class ThreadPool(object):
    """
    A simple thread pool implementation for executing script tasks.
    """
    def __init__(self, number_threads, device):
        self.number_threads = number_threads
        self.device_threads = []
        self.device = device
        self.tasks_queue = Queue(number_threads)

        # Create and start the worker threads.
        for _ in xrange(0, number_threads):
            thread = Thread(target=self.apply_scripts)
            self.device_threads.append(thread)
        for thread in self.device_threads:
            thread.start()

    def apply_scripts(self):
        """
        The main loop for a worker thread.

        Fetches a task from the queue, gathers data, runs the script,
        and disseminates the results.
        """
        while True:
            # Blocks until a task is available.
            script, location, neighbours = self.tasks_queue.get()

            # A None task is a "poison pill" to signal termination.
            if neighbours is None and script is None:
                self.tasks_queue.task_done()
                return

            script_data = []
            # Block Logic: Gather data from neighbors and self.
            # This involves acquiring a lock on each device's data location,
            # which can lead to deadlocks.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                result = script.run(script_data)
                # Disseminate results by setting data, which also releases locks.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                self.device.set_data(location, result)

            self.tasks_queue.task_done()

    def submit_task(self, script, location, neighbours):
        """Adds a new script execution task to the queue."""
        self.tasks_queue.put((script, location, neighbours))

    def join_threads(self):
        """Shuts down the thread pool cleanly."""
        # Wait for all pending tasks to be completed.
        self.tasks_queue.join()

        # Send a "poison pill" to each worker thread to make it exit.
        for _ in xrange(0, len(self.device_threads)):
            self.submit_task(None, None, None)

        # Wait for all worker threads to terminate.
        for thread in self.device_threads:
            thread.join()
