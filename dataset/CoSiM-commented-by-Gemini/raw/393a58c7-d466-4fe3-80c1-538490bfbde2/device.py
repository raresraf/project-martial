"""
Models a device in a distributed simulation environment.

This module defines the behavior of a `Device` in a simulated network, likely for
parallel data processing or sensor network simulations. It uses a multi-threaded
approach to handle device setup, script execution, and synchronization between
devices. The architecture relies on a master device to bootstrap synchronization
primitives (barriers, locks) for a group of devices, which then execute scripts
in synchronized time steps.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """Represents a node in the distributed simulation.

    Each device has its own data, can be assigned scripts to execute, and
    communicates with its neighbors. One device in a group acts as a "master"
    to coordinate the initialization of synchronization objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data,
                                keyed by location.
            supervisor (object): A supervisor object that provides network context,
                                 such as neighbor information.
        """
        # Synchronization events
        self.are_locks_ready = Event()
        self.script_received = Event()
        self.timepoint_done = Event()
        self.master_barrier = Event()

        # State and configuration
        self.master_id = None
        self.is_master = True
        self.barrier = None
        self.stored_devices = []
        self.data_lock = [None] * 100  # Pool of locks for data locations
        self.lock = Lock()  # General purpose lock for device state
        self.started_threads = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        # The main thread for this device's lifecycle
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the device group, electing a master and sharing sync objects.

        This method implements the bootstrap logic. The first device to enter
        becomes the master, creates the barrier and data locks, and signals
        that they are ready. Other "worker" devices wait for the master to
        complete this setup and then copy the references to these sync objects.

        Args:
            devices (list): A list of all Device objects in the group.
        """
        # Determine if another device has already become the master.
        for device in devices:
            if device is not None and device.master_id is not None:
                self.master_id = device.master_id
                self.is_master = False
                break

        if self.is_master is True:
            # This device becomes the master.
            self.barrier = ReusableBarrierSem(len(devices))
            self.master_id = self.device_id
            for i in range(100):
                self.data_lock[i] = Lock()
            # Signal that locks and barrier are created.
            self.are_locks_ready.set()
            self.master_barrier.set()
            for device in devices:
                if device is not None:
                    # Distribute the barrier to all other devices.
                    device.barrier = self.barrier
                    self.stored_devices.append(device)
        else:
            # This is a worker device.
            for device in devices:
                if device is not None:
                    if device.device_id == self.master_id:
                        # Wait for the master to set up sync objects.
                        device.master_barrier.wait()
                        if self.barrier is None:
                            # Copy the barrier reference from the master.
                            self.barrier = device.barrier
                    self.stored_devices.append(device)

    def assign_script(self, script, location):
        """Assigns a script to be executed at a specific data location.

        This method is called by an external entity to give work to the device.
        If a script is provided, it's added to the work queue and the main
        device thread is notified. If `script` is None, it signals the end of a
        timepoint.

        Args:
            script (object): The script object with a `run` method.
            location (int): The data location index for the script to operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Ensure the master device has finished setting up locks before proceeding.
            for device in self.stored_devices:
                if device.device_id == self.master_id:
                    device.are_locks_ready.wait()
            # Get a reference to the shared data locks from the master.
            for device in self.stored_devices:
                if device.device_id == self.master_id:
                    self.data_lock = device.data_lock
            self.script_received.set()
        else:
            # A None script signifies the end of script assignments for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Atomically sets data at a specific sensor location."""
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main long-running thread that orchestrates a device's operations."""

    def __init__(self, device):
        """Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """The main lifecycle loop of the device.

        This loop represents the progression of time in the simulation. In each
        iteration (timepoint), it waits for scripts, executes them, and then
        synchronizes with all other devices at a barrier before starting the
        next timepoint.
        """
        while True:
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown.
                break

            # Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # For each assigned script, create and start an ExecutorThread.
            for (script, location) in self.device.scripts:
                executor = ExecutorThread(self.device, script, neighbours, location)
                self.device.started_threads.append(executor)
                executor.start()

            # Wait for all executor threads for this timepoint to complete.
            for executor in self.device.started_threads:
                executor.join()

            # Clean up for the next timepoint.
            del self.device.started_threads[:]
            self.device.timepoint_done.clear()
            # Synchronize with all other devices. No device proceeds to the next
            # timepoint until all have reached this barrier.
            self.device.barrier.wait()


class ExecutorThread(Thread):
    """A short-lived thread to execute a single script for one timepoint."""

    def __init__(self, device, script, neighbours, location):
        """Initializes the script executor thread."""
        Thread.__init__(self, name="Executor Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.location = location

    def run(self):
        """Executes the script.

        The core logic of the simulation:
        1. Acquire a lock for the specific data location to prevent race conditions.
        2. Gather data from itself and its neighbors at that location.
        3. Run the script on the collected data.
        4. Write the result back to itself and its neighbors.
        5. Release the lock.
        """
        self.device.data_lock[self.location].acquire()

        if self.neighbours is None:
            return

        script_data = []
        # Gather data from all neighboring devices for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Include its own data.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Run the computational script on the aggregated data.
            result = self.script.run(script_data)
            # Propagate the result back to all neighbors.
            for device in self.neighbours:
                device.set_data(self.location, result)
            # Update its own data with the result.
            self.device.set_data(self.location, result)

        # Release the lock for the data location.
        self.device.data_lock[self.location].release()
