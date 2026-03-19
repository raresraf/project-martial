"""
This module defines the core components for a simulated device in a distributed,
time-stepped environment, likely for sensor network or multi-agent simulations.

It includes the Device class, which represents an individual node, and the
DeviceThread class, which manages the execution logic for that node using a
thread pool.
"""


from threading import Event, Thread, Lock
from Queue import Queue
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device or node within the simulation.

    Each device has a unique ID, local sensor data, and a connection to a
    central supervisor. It manages its state through threading events and
    synchronizes with other devices using locks and barriers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data,
                                keyed by location.
            supervisor (Supervisor): The central supervisor object that manages the
                                     simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been assigned.
        self.script_received = Event()
        # A list to hold assigned scripts to be executed in the current time step.
        self.scripts = []
        # Event to signal that all scripts for a time point have been assigned.
        self.timepoint_done = Event()

        # Synchronization primitives.
        self.barrier = None
        # Locks for thread-safe access to sensor_data locations.
        self.locks = {}

        # The main worker thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()


    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes synchronization objects for the device and its peers.

        This method sets up locks for local data and, for device 0, creates the
        shared barrier for all devices in the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Pre-condition: Initialize a lock for each data location to ensure thread-safe access.
        for loc in self.sensor_data:
            self.locks[loc] = Lock()

        # Invariant: Device 0 acts as the master for barrier creation.
        if self.device_id == 0:
            # The barrier ensures all devices complete a time step before starting the next.
            self.barrier = ReusableBarrierCond(len(devices))

            # Post-condition: Distribute the shared barrier to all other devices.
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        If a script is provided, it's added to the queue and the `script_received`
        event is set. If `script` is None, it signals the end of the timepoint.

        Args:
            script (Script): The script object to execute.
            location (str): The data location the script applies to.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script is a sentinel value to indicate the end of script assignments for this tick.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safely retrieves data from a specific location.

        Args:
            location (str): The key for the desired data.

        Returns:
            The data associated with the location, or None if the location is not found.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Thread-safely updates data at a specific location.

        Args:
            location (str): The key for the data to update.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """Joins the main device thread to gracefully shut down the device."""
        self.thread.join()


class DeviceThread(Thread):
    """
    A worker thread that manages script execution for a single Device.

    This thread maintains a pool of sub-threads to process scripts concurrently,
    using a queue to distribute work.
    """

    def __init__(self, device):
        """
        Initializes the main thread for a device.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = []
        # Queue for distributing script execution tasks to the thread pool.
        self.queue = Queue()
        self.create_threads()

    def create_threads(self):
        """Creates and starts a pool of worker threads."""
        # Create a fixed-size thread pool to execute script tasks.
        for _ in xrange(8):


            thread = Thread(target=self.work)
            self.threads.append(thread)

        for thread in self.threads:
            thread.start()

    def join_threads(self):
        """Signals worker threads to terminate and waits for them to complete."""
        # Pre-condition: Put sentinel values (None) on the queue to signal termination
        # to each worker thread.
        for _ in xrange(8):
            self.queue.put((None, None, None))

        # Block until all items in the queue have been processed.
        self.queue.join()

        # Post-condition: Join all worker threads to ensure they have exited cleanly.
        for thread in self.threads:
            thread.join()

    def work(self):
        """
        The target function for worker threads in the pool.

        Pulls a script task from the queue, gathers necessary data from neighbor
        devices, executes the script, and updates data on all affected devices.
        """
        while True:
            # Block until a task is available on the queue.
            script, location, neighbours = self.queue.get()

            # Sentinel check: A None script signals thread termination.
            if script is None:
                self.queue.task_done()
                break

            script_data = []

            # Block Logic: Gather data from all neighboring devices for the specified location.
            # Pre-condition: `neighbours` contains the list of devices to query.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

            # Also gather this device's own data for the location.
            data = self.device.get_data(location)

            if data is not None:
                script_data.append(data)

            # Invariant: Execute the script only if there is data to process.
            if script_data != []:
                # Run the script with the aggregated data.
                result = script.run(script_data)

                # Post-condition: Propagate the script's result back to all neighbors and self.
                for device in neighbours:


                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)

                self.device.set_data(location, result)

            self.queue.task_done()

    def run(self):
        """
        The main execution loop for the DeviceThread.

        This loop represents the device's lifecycle through the simulation's time steps.
        """
        # Loop indefinitely, processing one time step per iteration.
        while True:
            # Get the current set of neighbors from the supervisor for this time step.
            neighbours = self.device.supervisor.get_neighbours()

            # A None value for neighbors signals the end of the simulation.
            if neighbours is None:
                break
            # Inner loop to process all scripts for the current time step.
            while True:

                # Pre-condition: Check if new scripts have been received.
                if self.device.script_received.isSet():

                    # Clear the event and process the scripts.
                    self.device.script_received.clear()

                    for (script, location) in self.device.scripts:
                        self.queue.put((script, location, neighbours))

                # Pre-condition: Check if the supervisor has signaled the end of script assignment
                # for this time step.
                if self.device.timepoint_done.isSet():
                    self.device.timepoint_done.clear()
                    # Reset the script received flag for the next time step.
                    self.device.script_received.set()
                    # Exit the inner loop to proceed to synchronization.
                    break

            # Block until all scripts in the queue for this time step are processed.
            self.queue.join()

            # Invariant: All devices must wait at the barrier. This ensures that all devices
            # have completed their computations for the current time step before any
            # device proceeds to the next one, maintaining simulation consistency.
            self.device.barrier.wait()

        # Post-condition: After the main simulation loop ends, join all worker threads.
        self.join_threads()
