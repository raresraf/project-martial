"""
This module implements a multi-threaded device simulation framework with a
two-level threading architecture.

- A `Device` object represents a node in the network.
- Each `Device` has one main `DeviceThread`.
- Each `DeviceThread` acts as a dispatcher, managing its own internal pool of
  `Worker` threads (7 workers) to process tasks in parallel.
- Synchronization is handled via a mix of barriers, locks, semaphores, and events
  to coordinate work within a device and across all devices in the simulation.
- Data consistency is maintained through a global, per-location locking mechanism.
"""

from threading import Event, Thread, Lock, Semaphore
# A custom barrier implementation is expected.
from barrier import ReusableBarrierCond


class Device(object):
    """Represents a device node in the simulation.

    This class holds the device's state (ID, sensor data) and manages the main
    `DeviceThread` that drives its operation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (object): The central supervisor for the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Signals arrival of new scripts.
        self.scripts = []
        self.scripts_lock = Lock()
        self.timepoint_done = Event()  # Signals the end of a timepoint's script assignments.
        self.barrier = None
        self.location_locks = {}  # Shared, global locks for each data location.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources across all devices.

        This method should be called on one "main" device. It initializes and
        distributes a shared barrier and a set of per-location locks to all
        devices in the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        barrier = ReusableBarrierCond(len(devices))
        for device in devices:
            device.barrier = barrier

        # Create a global lock for each unique data location.
        location_locks = {}
        for device in devices:
            for location in device.sensor_data:
                if location not in location_locks:
                    location_locks[location] = Lock()

        # Distribute the location locks to all devices.
        for device in devices:
            device.location_locks = location_locks

    def assign_script(self, script, location):
        """Assigns a new script to the device.

        A `None` script is a special signal indicating the end of a timepoint.
        """
        if script is not None:
            self.scripts_lock.acquire()
            self.scripts.append((script, location))
            self.scripts_lock.release()
            self.script_received.set()


        else:
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """Gets sensor data for a location. Caller must handle locking."""
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None

    def set_data(self, location, data):
        """Sets sensor data for a location. Caller must handle locking."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def process_work(self, script, location, neighbours):
        """Processes a single script in a thread-safe manner.

        This method acquires the lock for the specified location, gathers data
        from neighbors, runs the script, and distributes the results.
        """
        self.location_locks[location].acquire()

        script_data = []

        # Gather data from neighbors.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Gather data from the local device itself. This is redundant if
        # the device is included in its own neighbors list.
        data = self.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = script.run(script_data)

            for device in neighbours:
                device.set_data(location, result)

            self.set_data(location, result)

        self.location_locks[location].release()

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main thread for a device, which manages a pool of worker threads."""

    def __init__(self, device):
        """Initializes the main thread for a device."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device.

        This loop coordinates the device's internal worker pool through
        simulation timepoints, synchronizing with other devices via a barrier.
        """
        # --- Worker Pool Setup ---
        work_lock = Lock()
        work_pool_empty = Event()
        work_pool_empty.set()
        work_pool = []
        workers = []
        workers_number = 7
        work_available = Semaphore(0)
        own_work = None

        for worker_id in range(1, workers_number + 1):
            workers.append(Worker(worker_id, work_pool, work_available, work_pool_empty, work_lock, self.device))
            workers[worker_id-1].start()

        # --- Main Simulation Loop ---
        while True:
            scripts_ran = []
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is not None:
                neighbours = set(neighbours)
                if self.device in neighbours:
                    neighbours.remove(self.device)
            else:
                # Supervisor signals termination. Signal workers to exit.
                for i in range(workers_number):
                    work_available.release()
                for worker in workers:
                    worker.join()
                break

            # Sync with all other devices before starting the timepoint's work.
            self.device.barrier.wait()

            # --- Inner Loop: Process scripts for the current timepoint ---
            while True:
                self.device.script_received.wait()

                self.device.scripts_lock.acquire()

                # Dispatch scripts to the worker pool.
                for (script, location) in self.device.scripts:
                    if script in scripts_ran:
                        continue
                    scripts_ran.append(script)

                    # The main device thread handles one piece of work itself,
                    # and offloads the rest to its worker pool.
                    if own_work is None:
                        own_work = (script, location, neighbours)
                    else:
                        work_lock.acquire()
                        work_pool.append((script, location, neighbours))
                        work_pool_empty.clear()
                        work_available.release()
                        work_lock.release()

                self.device.scripts_lock.release()

                # Check if the timepoint has ended and all scripts have been dispatched.
                if self.device.timepoint_done.is_set() and len(scripts_ran) == len(self.device.scripts):
                    if own_work is not None:
                        script, location, neighbours = own_work
                        own_work = None
                        self.device.process_work(script, location, neighbours)

                    # Wait for all workers to complete.
                    work_pool_empty.wait()
                    for worker in workers:
                        worker.work_done.wait()

                    self.device.timepoint_done.clear()
                    
                    # Sync with all other devices to mark the end of the timepoint.
                    self.device.barrier.wait()
                    break


class Worker(Thread):
    """A worker thread that executes tasks from a shared pool."""

    def __init__(self, worker_id, work_pool, work_available, work_pool_empty, work_lock, device):
        """Initializes the worker."""
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.work_pool = work_pool
        self.work_available = work_available
        self.work_pool_empty = work_pool_empty
        self.work_lock = work_lock
        self.device = device
        self.work_done = Event()
        self.work_done.set()

    def run(self):
        """The main loop for the worker thread."""
        while True:
            self.work_available.acquire() # Wait for a task.
            self.work_lock.acquire()
            self.work_done.clear()

            if not self.work_pool:
                # An empty pool after being signaled means termination.
                self.work_lock.release()
                return

            script, location, neighbours = self.work_pool.pop(0)

            if not self.work_pool:
                self.work_pool_empty.set()

            self.work_lock.release()

            # Process the assigned work.
            self.device.process_work(script, location, neighbours)

            self.work_done.set()
