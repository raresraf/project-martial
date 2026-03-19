


"""
This module defines a distributed device simulation framework with a focus
on concurrent script execution and data synchronization. It includes classes
for `Device`, `DeviceThread` (orchestrating script execution), and
`WorkerThread` (executing individual scripts), utilizing threading
primitives like Events, Locks, and a ReusableBarrierSem for coordination.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single device or processing unit within a distributed system.
    Each device manages its own sensor data, executes scripts, and interacts
    with other devices through a supervisor. It employs a multi-threaded
    architecture for script execution and uses various synchronization
    primitives to ensure data consistency and proper operation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): Initial sensor data for the device,
                                mapping locations to data values.
            supervisor (Supervisor): The supervisor managing this device
                                     and its interactions with others.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that the current timepoint's tasks are completed.
        self.timepoint_done = Event() # This event seems unused in this version, `script_received` is used for timepoint completion.
        self.thread = DeviceThread(self)

        # List to hold references to other devices in the system.
        self.devices = []

        # List of locks, where each lock protects a specific data location.
        self.locations = []

        # Event to signal that the device setup (including barrier and location locks) is complete.
        self.setup_start = Event()

        # Locks for controlling concurrent access to data retrieval and setting.
        self.set_lock = Lock()
        self.get_lock = Lock()

        # Shared barrier for inter-device synchronization.
        self.barrier = None

        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the device with a list of other devices in the system
        and establishes shared synchronization primitives (barrier and location-specific locks).

        This method is primarily called by the "master" device (device_id 0)
        to initialize shared resources that all devices will use.

        Args:
            devices (list): A list of Device objects representing
                            all devices in the system.
        """
        self.devices = devices
        num_devices = len(devices)

        # Block Logic: Initializes a reusable barrier for synchronizing all devices.
        barrier = ReusableBarrierSem(num_devices)

        # Block Logic: This block is executed only by the device with device_id 0,
        # acting as a coordinator to set up shared resources for all devices.
        if self.device_id == 0:
            # Block Logic: Creates a list of locks, one for each potential data location,
            # to control concurrent access to sensor data.
            for _ in range(25): # Assuming 25 possible data locations based on context.
                lock = Lock()
                self.locations.append(lock)

            # Block Logic: Propagates the shared location locks and the barrier
            # to all other devices in the system, then signals them to start.
            for device in devices:
                device.locations = self.locations
                device.barrier = barrier
                device.setup_start.set() # Signals that setup is complete for this device.

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.

        Args:
            script (Script): The script object to be executed.
                             If None, it signifies the end of script assignments
                             for the current timepoint.
            location (int): The identifier for the data location
                            the script operates on.
        """
        # Block Logic: If a script is provided, it's added to the list of scripts
        # to be processed in the current timepoint.
        if script is not None:
            self.scripts.append((script, location))
        # Block Logic: If no script is provided, it signals that all scripts for
        # the current timepoint have been assigned, allowing the DeviceThread to proceed.
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location.

        This operation is protected by a lock to ensure thread-safe access
        to the sensor data.

        Args:
            location (int): The identifier of the data location.

        Returns:
            Any: The sensor data at the given location, or None if the
                 location does not exist in the device's sensor_data.
        """
        # Block Logic: Acquires a lock to ensure exclusive read access to sensor data.
        with self.get_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location.

        This operation is protected by a lock to ensure thread-safe modification
        of the sensor data.

        Args:
            location (int): The identifier of the data location to update.
            data (Any): The new data value to set for the location.
        """
        # Block Logic: Acquires a lock to ensure exclusive write access to sensor data.
        with self.set_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device,
        waiting for its background thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    A dedicated thread for a Device object, responsible for orchestrating
    script execution by dispatching tasks to a pool of `WorkerThread` instances.
    It manages timepoint synchronization, neighbor discovery, and gracefully
    handles device shutdown.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.

        This method waits for the device setup to complete, then continuously
        monitors for new timepoints, retrieves neighbor information,
        dispatches scripts to `WorkerThread` instances, and handles
        synchronization across all devices.
        """
        # Block Logic: Waits for the device's initial setup to be completed
        # by the coordinating device (device_id 0).
        self.device.setup_start.wait()

        # Block Logic: Main loop for the device thread, continuously processing
        # timepoints until a shutdown signal (None neighbors) is received.
        while True:
            # Precondition: The supervisor provides the current neighbors for the device.
            # Invariant: If neighbours are None, it signals the thread to terminate.
            neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: If neighbours are None, it indicates a shutdown request.
            if neighbours is None:
                break

            # Block Logic: Waits for a new script to be assigned for the current timepoint
            # and then clears the event for the next timepoint.
            self.device.script_received.wait()
            self.device.script_received.clear()

            index = 0
            workers = []
            num_scripts = len(self.device.scripts)

            # Block Logic: Creates a WorkerThread for each assigned script.
            for _ in self.device.scripts:
                worker = WorkerThread(self.device, neighbours, index)
                workers.append(worker)
                index += 1

            # Block Logic: If the number of scripts is small (less than 8),
            # all workers are started and joined directly.
            if num_scripts < 8:
                for worker in workers:
                    worker.start()

                for worker in workers:
                    worker.join()
            # Block Logic: If there are 8 or more scripts, workers are started
            # and joined in batches of 8 for concurrent processing.
            else:
                aux = 0 # Auxiliary counter for tracking script batches.
                # Invariant: The loop continues until all scripts have been processed.
                while True:

                    # Block Logic: Breaks the loop if all scripts have been processed.
                    if num_scripts == 0:
                        break

                    # Block Logic: Processes scripts in batches of 8.
                    if num_scripts >= 8:
                        start = aux
                        end = aux + 8

                        # Block Logic: Starts workers for the current batch.
                        for i in range(start, end):
                            workers[i].start()

                        # Block Logic: Waits for workers in the current batch to complete.
                        for i in range(start, end):
                            workers[i].join()

                        aux += 8
                        num_scripts -= 8

                    # Block Logic: Processes the remaining scripts if less than 8.
                    elif num_scripts < 8:
                        start = aux
                        end = aux + num_scripts

                        # Block Logic: Starts workers for the remaining scripts.
                        for i in range(start, end):
                            workers[i].start()

                        # Block Logic: Waits for workers to complete and then breaks.
                        for i in range(start, end):
                            workers[i].join()
                        break

            # Block Logic: Waits at the shared barrier to synchronize with all other
            # devices before proceeding to the next timepoint.
            self.device.barrier.wait()


class WorkerThread(Thread):
    """
    A dedicated thread for executing a single assigned script on a device.

    It retrieves necessary data from the device and its neighbors for a specific
    script and location, runs the script, and then propagates the results back
    to the devices, ensuring thread-safe access to data locations.
    """

    def __init__(self, device, neighbours, index):
        """
        Initializes a new WorkerThread instance.

        Args:
            device (Device): The Device object on which the script will run.
            neighbours (list): A list of neighboring Device objects.
            index (int): The index of the script in the device's scripts list
                         that this worker is responsible for executing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.index = index

    def run(self):
        """
        Executes the assigned script.

        This method retrieves data from neighboring devices and the local
        device for the specific location associated with the script, runs
        the script with this aggregated data, and then updates the
        corresponding data locations on both the neighbors and the local device.
        Synchronization is ensured via location-specific locks.
        """
        # Block Logic: Retrieves the script and its associated location using the worker's index.
        (script, location) = self.device.scripts[self.index]

        # Block Logic: Acquires a lock for the current data location to ensure
        # exclusive access during data retrieval, script execution, and data update.
        self.device.locations[location].acquire()

        script_data = []

        # Block Logic: Gathers sensor data from all known neighboring devices
        # for the specified location.
        for neighbour in self.neighbours:
            data_neigh = neighbour.get_data(location)

            if data_neigh is not None:
                script_data.append(data_neigh)

        # Block Logic: Gathers sensor data from the local device for the specified location.
        own_data = self.device.get_data(location)
        if own_data is not None:
            script_data.append(own_data)

        # Precondition: There is collected data for the script to process.
        # Invariant: If data is present, the script is run and its results are
        # propagated to the devices.
        if script_data:
            # Block Logic: Executes the script with the collected data.
            result = script.run(script_data)

            # Block Logic: Updates the sensor data on all neighboring devices
            # with the result of the script execution.
            for neighbour in self.neighbours:
                neighbour.set_data(location, result)

            # Block Logic: Updates the sensor data on the local device
            # with the result of the script execution.
            self.device.set_data(location, result)

        # Block Logic: Releases the lock for the data location,
        # allowing other threads to access it.
        self.device.locations[location].release()

