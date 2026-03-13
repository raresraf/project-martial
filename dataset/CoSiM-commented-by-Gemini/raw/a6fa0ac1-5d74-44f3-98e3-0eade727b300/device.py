"""
This module defines a simulated Device using a complex master-worker threading
model for a distributed sensor network simulation.

The architecture involves a `Device` class that spawns a `DeviceThread` (master).
This master thread dynamically creates a pool of `Worker` threads and manages a
shared task pool. Synchronization is handled by a combination of a global
barrier, instance-level locks, and multiple events to coordinate the state
between the master and worker threads.
"""

from threading import Event, Thread, Lock
# Assuming barrier.py contains a ReusableBarrier implementation.
from barrier import ReusableBarrier


class Device(object):
    """
    Represents a single device, holding sensor data and managing its lifecycle thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor data.
            supervisor (Supervisor): The central supervisor controlling the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []  # Holds (script, location) tuples for the current timepoint.
        self.timepoint_done = Event() # Signals that script assignment is complete.
        self.thread = DeviceThread(self)
        self.locks = {} # A dictionary mapping locations to Lock objects.

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared locks and the global barrier for all devices.
        
        This method should be called once at the start of the simulation.
        """
        # Initialize a lock for each data location this device is aware of.
        for key in self.sensor_data:
            self.locks[key] = Lock()

        # The device with ID 0 is responsible for creating and distributing the shared barrier.
        if self.device_id == 0:
            self.thread.barrier = ReusableBarrier(len(devices))

            for device in devices:
                device.thread.barrier = self.thread.barrier

            # Start all device threads after the barrier is set up.
            for device in devices:
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of assignment.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def has_data(self, location):
        """Checks if the device tracks data for a given location."""
        return location in self.sensor_data

    def get_data(self, location):
        """
        Acquires a lock and returns data for a given location.

        NOTE: This implementation has a potential flaw. The lock is acquired here
        but released in `set_data`. This means the lock is held by a worker
        thread for the entire duration of data aggregation, script execution,
        and data propagation, which can be inefficient and lead to deadlocks
        if not managed carefully.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates data for a given location and releases the lock.
        
        NOTE: This method is coupled with `get_data` for lock management.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """Joins the main device thread to shut down gracefully."""
        self.thread.join()


class DeviceThread(Thread):
    """The master thread that manages a pool of worker threads for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = None
        # --- Worker Pool Management ---
        self.work_pool_lock = Lock() # Protects access to the work_pool list.
        self.work_pool_empty = Event() # Signals that the work_pool is empty.
        self.work_ready = Event() # Signals that there are tasks in the work_pool.
        self.work_pool = [] # A list acting as a task queue for workers.
        self.simulation_complete = False # Flag to terminate worker threads.

        self.work_ready.clear()
        self.work_pool_empty.set()

    def run(self):
        """The main control loop for the master thread."""
        workers = []

        while True:
            # Get neighbors at the start of each timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation signal.

            # Wait for the supervisor to finish assigning scripts.
            self.device.timepoint_done.wait()

            # --- Worker Pool Scaling ---
            # Dynamically create worker threads if needed, up to a max of 8.
            for i in range(len(workers), len(self.device.scripts)):
                if len(workers) < 8:
                    worker = Worker(self.work_pool_empty, self.work_ready, self.work_pool_lock, self)
                    workers.append(worker)
                    worker.start()

            # Synchronize with all other devices before distributing tasks.
            self.barrier.wait()

            # --- Task Creation and Distribution ---
            for (script, location) in self.device.scripts:
                script_devices = [d for d in neighbours if d.has_data(location)]

                if script_devices:
                    if self.device not in script_devices:
                        script_devices.append(self.device)
                    script_devices.sort(key=lambda x: x.device_id)
                    
                    # Safely add the new task to the shared work pool.
                    with self.work_pool_lock:
                        self.work_pool.append(Task(script, location, script_devices))
                        self.work_ready.set() # Notify workers that a task is ready.
                        self.work_pool_empty.clear()

            # Wait for the work pool to become empty.
            self.work_pool_empty.wait()
            # Also ensure all workers have finished their current task.
            for worker in workers:
                worker.work_done.wait()
            
            # Cleanup for the next timepoint.
            self.device.scripts = []
            self.device.timepoint_done.clear()

        # --- Shutdown ---
        # Signal workers to terminate and wait for them to exit.
        with self.work_pool_lock:
            self.simulation_complete = True
            self.work_ready.set() # Wake up any waiting workers.

        for worker in workers:
            worker.join()


class Worker(Thread):
    """A worker thread that processes tasks from a shared pool."""

    def __init__(self, work_pool_empty, work_ready, work_pool_lock, device_thread):
        Thread.__init__(self, name="Worker Thread")
        self.work_pool_lock = work_pool_lock
        self.work_pool_empty = work_pool_empty
        self.work_ready = work_ready
        self.device_thread = device_thread
        self.work_done = Event() # Signals that this worker has finished its current task.
        self.work_done.set()

    def run(self):
        """The main loop for a worker thread."""
        while True:
            self.work_ready.wait() # Wait for the master to signal that tasks are available.
            if self.device_thread.simulation_complete:
                break # Exit if the simulation is over.

            with self.work_pool_lock:
                if not self.device_thread.work_pool:
                    # If the pool is empty, set the empty event and prepare to wait again.
                    self.work_pool_empty.set()
                    if not self.device_thread.simulation_complete:
                        self.work_ready.clear()
                    continue
                else:
                    # Pop a task from the shared pool.
                    self.work_done.clear()
                    task = self.device_thread.work_pool.pop(0)

            # --- Task Execution ---
            # The lock is released here, allowing other workers to access the pool.
            data = [device.get_data(task.location) for device in task.devices]
            result = task.script.run(data)
            for device in task.devices:
                device.set_data(task.location, result)

            # Signal that this specific worker's task is done.
            self.work_done.set()


class Task(object):
    """A simple data class to encapsulate a task for a worker."""
    def __init__(self, script, location, devices):
        self.devices = devices
        self.script = script
        self.location = location
