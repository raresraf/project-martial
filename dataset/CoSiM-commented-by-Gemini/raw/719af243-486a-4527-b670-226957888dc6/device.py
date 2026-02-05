"""
This module defines the core components for a distributed, parallel device simulation.

It provides a framework for simulating a network of devices that can execute
computational scripts on shared data in synchronized time steps. The architecture
relies heavily on Python's threading capabilities to manage concurrent
operations and synchronization between devices.
"""


from threading import Event, Thread, RLock
from Queue import Queue
from barrier import ReusableBarrier

class Device(object):
    """Represents a single device or node in the simulation network.

    Each device has a unique ID, its own local sensor data, and a reference to a
    central supervisor. It manages a list of scripts to be executed and uses a
    dedicated DeviceThread to control its execution lifecycle.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor data,
                                keyed by location.
            supervisor (Supervisor): An object that manages the overall simulation and
                                     device interactions.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []

        # Event to signal that the current simulation timepoint is complete.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # Dictionary of re-entrant locks, one for each data location.
        self.locations_locks = {}
        # A shared barrier for synchronizing all devices at the end of a timepoint.
        self.reusable_barrier = None

    def __str__(self):
        """Provides a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources for a collection of devices.

        This method, intended to be called on a single device (e.g., device 0),
        initializes and distributes shared synchronization primitives (locks and
        a reusable barrier) to all devices in the simulation.

        Args:
            devices (list[Device]): A list of all device objects in the simulation.
        """
        if self.device_id != 0:
            return
        locations_locks = {}
        reusable_barrier = ReusableBarrier(len(devices))
        for device in devices:
            device.locations_locks = locations_locks
            device.reusable_barrier = reusable_barrier

    def assign_script(self, script, location):
        """Assigns a script to be executed at a specific location.

        This method is called by the supervisor to give work to the device.
        It also handles signaling the end of a timepoint when a None script
        is received.

        Args:
            script (Script): The script object to be executed. If None, it signals
                             that the current timepoint has ended.
            location (any): The location context for the script execution, used as
                            a key for data and locks.
        """

        # Lazily initialize a lock for a location if it's the first time we see it.
        if location not in self.locations_locks:
            self.locations_locks[location] = RLock()

        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is the signal that all scripts for this timepoint have been assigned.
            self.timepoint_done.set()

        self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location.

        Args:
            location (any): The key for the desired data.

        Returns:
            The data associated with the location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location.

        Args:
            location (any): The key for the data to be updated.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class WorkerThread(Thread):
    """A worker thread that executes a single script task from a queue.

    Each DeviceThread manages a pool of these workers to process scripts in parallel.
    """

    def __init__(self, master, index):
        """Initializes the WorkerThread.

        Args:
            master (DeviceThread): The parent DeviceThread that owns this worker.
            index (int): The numerical index of this worker thread.
        """
        Thread.__init__(self, name="Worker Thread %d" % index)
        self.master = master
        self.index = index

    def get_script_data(self, location, neighbours):
        """Aggregates data for a given location from this device and its neighbors.

        Args:
            location (any): The location for which to gather data.
            neighbours (list[Device]): A list of neighboring device objects.

        Returns:
            list: A list of data values from all relevant devices.
        """
        # Collect data from all neighbors that have data for the given location.
        data = [d for d in [n.get_data(location) for n in neighbours] if d is not None]

        # Include this device's own data if available.
        my_data = self.master.device.get_data(location)
        if my_data is not None:
            data.append(my_data)

        return data

    def broadcast_result(self, location, result, neighbours):
        """Writes the script's result back to this device and all its neighbors.

        Args:
            location (any): The location to update.
            result (any): The new data value to be written.
            neighbours (list[Device]): The list of neighboring devices to update.
        """
        self.master.device.set_data(location, result)
        for device in neighbours:
            device.set_data(location, result)

    def run(self):
        """The main loop for the worker thread.

        Continuously fetches tasks from the master's queue and executes them
        until a None task is received, which signals shutdown.
        """
        while True:
            # Block until a task is available in the queue.
            task = self.master.queue.get()
            if task is None:
                # Sentinel value for shutting down the worker thread.
                break

            script = task[0]
            location = task[1]
            neighbours = task[2]

            # Acquire the lock for the specific location to ensure thread-safe
            # access to the data at that location.
            with self.master.device.locations_locks[location]:
                script_data = self.get_script_data(location, neighbours)
                if script_data != []:
                    result = script.run(script_data)
                    self.broadcast_result(location, result, neighbours)

class DeviceThread(Thread):
    """The main control thread for a single Device.

    This thread manages the device's lifecycle, which is structured into discrete
    timepoints. It spawns a pool of WorkerThreads to handle parallel script
    execution and uses a ReusableBarrier to synchronize with all other devices
    at the end of each timepoint.
    """

    def __init__(self, device):
        """Initializes the DeviceThread.

        Args:
            device (Device): The device object that this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue()

    def run(self):
        """The main simulation loop for the device.

        The loop represents the progression of time in discrete steps (timepoints).
        In each step, it gets neighbors, dispatches assigned scripts to worker
        threads, and synchronizes with other devices.
        """
        # Create a pool of worker threads to execute script tasks.
        threads_number = 8
        workers = []
        for i in range(0, threads_number):
            workers.append(WorkerThread(self, i))

        # Start all worker threads.
        for worker in workers:
            worker.start()

        while True:
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None response from get_neighbours signals the end of the simulation.
                # Put sentinels in the queue to terminate all worker threads.
                for i in range(0, threads_number):
                    self.queue.put(None)
                break

            # Wait until all scripts for the current timepoint have been received.
            # This is signaled by the supervisor calling assign_script with a None script.
            while not self.device.timepoint_done.isSet():
                self.device.script_received.wait()
            self.device.script_received.clear()

            # Add all assigned scripts for this timepoint to the worker queue.
            for (script, location) in self.device.scripts:
                self.queue.put((script, location, neighbours))

            # --- Synchronization Point ---
            # All devices must wait here until every device has finished processing
            # its scripts for the current timepoint. This ensures that the state is
            # consistent across the network before proceeding to the next timepoint.
            self.device.reusable_barrier.wait()
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

        # Wait for all worker threads to complete their shutdown.
        for worker in workers:
            worker.join()
			