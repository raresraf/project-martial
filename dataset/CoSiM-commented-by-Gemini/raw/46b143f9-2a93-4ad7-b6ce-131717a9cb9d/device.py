"""
Models a device in a distributed, time-stepped simulation environment.

This module defines the classes for simulating a network of devices that
process data concurrently. The system appears to be designed for scenarios like
distributed sensing or parallel computation, where devices operate on shared data
at specific locations and must synchronize their operations in discrete time steps.

Classes:
    Device: Represents a single computational node in the network.
    DeviceThread: The main control thread for a Device, managing its lifecycle and work.
    Worker: A thread for executing a single computational task (a "script") for a Device.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device in a simulated distributed network.

    Each device has an ID, local sensor data, and a connection to a supervisor.
    It executes scripts on data, coordinating with neighboring devices. It uses
    a pool of worker threads to process scripts concurrently.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor values.
            supervisor (Supervisor): A reference to the central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Event to signal that a new script has been assigned.
        self.script_received = Event()
        self.scripts = []
        self.scripts_lock = Lock()
        
        # Event to signal the end of a timepoint's work.
        self.timepoint_done = Event()
        
        # Synchronization barrier for all devices in the simulation.
        self.barrier = None
        
        # Locks to ensure exclusive access to data at a specific location.
        self.location_locks = {}
        
        # The main thread that runs the device's lifecycle.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for a set of devices.

        This method sets up a shared barrier for synchronization and creates a
        set of locks for all unique data locations across all devices.

        Args:
            devices (list[Device]): The list of all devices in the simulation.
        """
        # A barrier to synchronize all devices at each time step.
        barrier = ReusableBarrierCond(len(devices))
        for device in devices:
            device.barrier = barrier

        location_locks = {}

        # Create a unique lock for each sensor location to prevent race conditions.
        for device in devices:
            for location in device.sensor_data:
                if location not in location_locks:
                    location_locks[location] = Lock()

        for device in devices:
            device.location_locks = location_locks

    def assign_script(self, script, location):
        """
        Assigns a computational script to this device for a given timepoint.

        Args:
            script (Script): The script object to be executed.
            location (any): The data location the script will operate on.
        """
        if script is not None:
            # Add the script to the list of work to be done.
            self.scripts_lock.acquire()
            self.scripts.append((script, location))
            self.scripts_lock.release()
            self.script_received.set()
        else:
            # A None script signals that there is no more work for this timepoint.
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location from this device.

        Args:
            location (any): The location to query for data.

        Returns:
            The data at the given location, or None if the location is not
            known to this device.
        """
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specific location on this device.

        Args:
            location (any): The location to update.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def process_work(self, script, location, neighbours):
        """
        Executes a script on data from a specific location.

        This is the core data processing function. It gathers data from this
        device and its neighbors at a given location, runs the script on the
        combined data, and then propagates the result back to all involved devices.
        Access to the location is serialized via a lock.

        Args:
            script (Script): The script to execute.
            location (any): The location of the data to process.
            neighbours (list[Device]): A list of neighboring devices to coordinate with.
        """
        self.location_locks[location].acquire()

        script_data = []

        # Gather data from all neighboring devices for the specified location.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Include this device's own data.
        data = self.get_data(location)
        if data is not None:
            script_data.append(data)

        # Run the script and update the data on all participating devices.
        if script_data:
            result = script.run(script_data)

            for device in neighbours:
                device.set_data(location, result)

            self.set_data(location, result)

        self.location_locks[location].release()

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a Device, managing its lifecycle."""

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main event loop for the device.

        This loop coordinates the device's participation in the time-stepped
        simulation. It synchronizes with other devices, manages a pool of
        worker threads, and distributes assigned scripts for execution.
        """
        work_lock = Lock()
        work_pool_empty = Event()
        work_pool_empty.set()
        work_pool = []
        workers = []
        workers_number = 7
        work_available = Semaphore(0)
        own_work = None

        # Create and start a pool of worker threads.
        for worker_id in range(1, workers_number + 1):
            workers.append(Worker(worker_id, work_pool, work_available, work_pool_empty, work_lock, self.device))
            workers[worker_id-1].start()

        while True:
            scripts_ran = []
            
            # Get the set of neighbors for the current context.
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is not None:
                neighbours = set(neighbours)
                if self.device in neighbours:
                    neighbours.remove(self.device)

            if neighbours is None:
                # A None for neighbors is the shutdown signal.
                # Release the semaphore to unblock all workers so they can terminate.
                for i in range(0,7):
                    work_available.release()

                # Wait for all worker threads to finish.
                for worker in workers:
                    worker.join()
                break

            # --- Start of a timepoint ---
            # Synchronize with all other devices before starting the timepoint.
            self.device.barrier.wait()

            while True:
                # Wait until a script is assigned or the timepoint is marked as done.
                self.device.script_received.wait()

                self.device.scripts_lock.acquire()

                # Process all newly received scripts.
                for (script, location) in self.device.scripts:
                    # Avoid running the same script multiple times in one timepoint.
                    if script in scripts_ran:
                        continue

                    scripts_ran.append(script)

                    # The device's own thread processes the first script to save thread-switching overhead.
                    # Subsequent scripts are offloaded to the worker pool.
                    if own_work is None:
                        own_work = (script, location, neighbours)
                    else:
                        work_lock.acquire()
                        work_pool.append((script, location, neighbours))
                        work_pool_empty.clear()
                        work_available.release()
                        work_lock.release()

                self.device.scripts_lock.release()

                # Check if all work for the current timepoint has been assigned and processed.
                if self.device.timepoint_done.is_set() and len(scripts_ran) == len(self.device.scripts):
                    
                    # Process the work assigned to this main thread, if any.
                    if own_work is not None:
                        script, location, neighbours = own_work
                        own_work = None
                        self.device.process_work(script, location, neighbours)

                    # Wait for the worker pool to finish all its tasks.
                    work_pool_empty.wait()

                    # Ensure each individual worker has finished its current task.
                    for worker in workers:
                        worker.work_done.wait()

                    self.device.timepoint_done.clear()
                    
                    # --- End of a timepoint ---
                    # Synchronize with all other devices before proceeding to the next timepoint.
                    self.device.barrier.wait()
                    break


class Worker(Thread):
    """
    A worker thread that executes a single computational script for a Device.
    """

    def __init__(self, worker_id, work_pool, work_available, work_pool_empty, work_lock, device):
        """
        Initializes the Worker thread.

        Args:
            worker_id (int): A unique ID for the worker.
            work_pool (list): The shared pool of tasks.
            work_available (Semaphore): A semaphore to signal available work.
            work_pool_empty (Event): An event to signal when the work pool is empty.
            work_lock (Lock): A lock to protect access to the work pool.
            device (Device): The parent device this worker belongs to.
        """
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
            # Wait for a task to become available.
            self.work_available.acquire()
            self.work_lock.acquire()
            self.work_done.clear()

            # The shutdown signal is an empty work pool.
            if not self.work_pool:
                self.work_lock.release()
                return

            # Pop a task from the work pool.
            script, location, neighbours = self.work_pool.pop(0)

            # If the pool is now empty, set the event.
            if not self.work_pool:
                self.work_pool_empty.set()

            self.work_lock.release()
            
            # Process the task.
            self.device.process_work(script, location, neighbours)
            
            # Signal that this worker is finished with its current task.
            self.work_done.set()