


"""
This module defines a distributed device simulation framework, featuring
`Device`, `DeviceThread`, and `DeviceWorkerThread` classes. It manages
device-specific data, script execution across locations, inter-device
communication, and synchronization using `threading` primitives like
Events, Locks, and a custom `ReusableBarrierCond`.
"""

from threading import Event, Thread, Condition, Lock

class Device(object):
    """
    Represents a single device or processing unit within a distributed system.
    Each device manages its own sensor data, executes scripts (potentially
    multiple per location), and interacts with other devices through a supervisor.
    It employs a multi-threaded architecture with fine-grained locking and
    a barrier for timepoint synchronization.
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
        # Event to signal when all scripts for the current timepoint have been assigned.
        self.scripts_received = Event()
        # Dictionary to store lists of scripts, keyed by location.
        self.scripts_dict = {}
        # Dictionary to store locks, keyed by location, for fine-grained data access control.
        self.locations_locks = {}
        # Barrier for synchronizing all devices at the end of each timepoint.
        self.timepoint_done = None
        # Stores the current list of neighboring devices.
        self.neighbours = None
        self.thread = DeviceThread(self)
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

        This method is designed to ensure all devices share the same barrier
        and that locks for data locations are correctly propagated.

        Args:
            devices (list): A list of Device objects representing
                            all devices in the system.
        """
        nr_devices = len(devices)
        # Block Logic: Initializes a shared barrier if it hasn't been set yet for this device.
        # The barrier is then propagated to all other devices that don't have it.
        if self.timepoint_done is None:
            self.timepoint_done = ReusableBarrierCond(nr_devices)
            # Invariant: All devices in the system will be assigned the same
            # timepoint_done barrier for synchronization.
            for device in devices:
                if device.timepoint_done is None and device != self:
                    device.timepoint_done = self.timepoint_done

        # Block Logic: Ensures that all devices share the same lock objects for each data location.
        # If a lock for a location is created on one device, it's reused by others.
        for location in self.sensor_data.keys():
            if location not in self.locations_locks:
                self.locations_locks[location] = Lock() # Creates a new lock for this location.
                # Propagates the newly created lock to other devices that don't have it for this location.
                for device in devices:
                    if location not in device.locations_locks and \
                        device != self:
                        device.locations_locks[location] = \
                            self.locations_locks[location]



    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.

        Scripts are appended to a list associated with their location,
        allowing multiple scripts to be run consecutively for the same location.

        Args:
            script (Script): The script object to be executed.
                             If None, it signifies that all scripts for the
                             current timepoint have been assigned.
            location (int): The identifier for the data location
                            the script operates on.
        """
        # Block Logic: If a script is provided, it's added to the list of scripts
        # for the specified location.
        if script is not None:
            # Pre-condition: Checks if there's an existing list of scripts for this location.
            if location in self.scripts_dict:
                self.scripts_dict[location].append(script) # Appends to existing list.
            else:
                self.scripts_dict[location] = [] # Creates a new list if none exists.
                self.scripts_dict[location].append(script) # Appends the first script.
        # Block Logic: If no script is provided (script is None), it signals that all scripts for
        # the current timepoint have been assigned, allowing the DeviceThread to proceed.
        else:
            self.scripts_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location.

        Args:
            location (int): The identifier of the data location.

        Returns:
            Any: The sensor data at the given location, or None if the
                 location does not exist in the device's sensor_data.
        """
        return self.sensor_data[location] \
            if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location.

        Args:
            location (int): The identifier of the data location to update.
            data (Any): The new data value to set for the location.
        """
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
    script execution by dispatching tasks to `DeviceWorkerThread` instances.
    It manages timepoint synchronization, neighbor discovery, and ensures
    all assigned scripts are processed.
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

        This method continuously retrieves neighbor information, waits for
        scripts to be assigned, dispatches `DeviceWorkerThread` instances
        for each script location, and synchronizes all devices at the end
        of each timepoint.
        """
        # Block Logic: Main loop for the device thread, continuously processing
        # timepoints until a shutdown signal (None neighbors) is received.
        while True:
            # Block Logic: Retrieves the current list of neighboring devices from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: If neighbours are None, it indicates a shutdown request, so the thread breaks its loop.
            if self.device.neighbours is None:
                break

            # Block Logic: Waits until all scripts for the current timepoint have been assigned.
            self.device.scripts_received.wait()

            # Block Logic: Creates and starts a DeviceWorkerThread for each location that has scripts assigned.
            threads = []
            for location in self.device.scripts_dict.keys():
                thread = DeviceWorkerThread(self.device, location)
                thread.start()
                threads.append(thread)

            # Block Logic: Waits for all DeviceWorkerThreads to complete their execution for the current timepoint.
            for thread in threads:
                thread.join()

            # Block Logic: Resets the scripts_received event for the next timepoint.
            self.device.scripts_received.clear()

            # Block Logic: Waits at the shared barrier to synchronize with all other devices
            # before proceeding to the next timepoint.
            self.device.timepoint_done.wait()

class DeviceWorkerThread(Thread):
    """
    A worker thread spawned by `DeviceThread` to execute all scripts
    associated with a specific data location for the current timepoint.
    It ensures thread-safe access to data through location-specific locks.
    """

    def __init__(self, device, location):
        """
        Initializes a new DeviceWorkerThread instance.

        Args:
            device (Device): The parent Device object.
            location (int): The specific data location this worker is
                            responsible for processing scripts for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.location = location

    def run(self):
        """
        Executes all assigned scripts for its specific location.

        This method iterates through each script associated with its location,
        retrieves data from neighboring devices and the local device, runs
        the script with this data, and then updates the corresponding data
        locations on both the neighbors and the local device.
        Synchronization is ensured via location-specific locks.
        """
        # Block Logic: Iterates through each script assigned to this worker's location.
        for script in self.device.scripts_dict[self.location]:

            # Block Logic: Acquires a lock for the current data location to ensure
            # exclusive access during data retrieval, script execution, and data update.
            self.device.locations_locks[self.location].acquire()

            script_data = []
            # Block Logic: Gathers sensor data from all known neighboring devices
            # for the specified location.
            for device in self.device.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Gathers sensor data from the local device for the specified location.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Precondition: There is collected data for the script to process.
            # Invariant: If data is present, the script is run and its results are
            # propagated to the devices.
            if script_data != []:
                # Block Logic: Executes the script with the collected data.
                result = script.run(script_data)
                
                # Block Logic: Updates the sensor data on all neighboring devices
                # with the result of the script execution.
                for device in self.device.neighbours:
                    device.set_data(self.location, result)
                
                # Block Logic: Updates the sensor data on the local device
                # with the result of the script execution.
                self.device.set_data(self.location, result)

            # Block Logic: Releases the lock for the data location,
            # allowing other threads to access it.
            self.device.locations_locks[self.location].release()


class ReusableBarrierCond(object):
    """
    Implements a reusable barrier using a `Condition` variable.

    This barrier allows a fixed number of threads to wait for each other
    at a synchronization point and then proceed together. It can be
    reused multiple times for subsequent synchronization points.
    """



    def __init__(self, num_threads):
        """
        Initializes a new ReusableBarrierCond instance.

        Args:
            num_threads (int): The number of threads that must reach the
                                barrier before any can proceed.
        """
        self.num_threads = num_threads
        # Current count of threads that have reached the barrier in the current phase.
        self.count_threads = self.num_threads
        # Condition variable used for blocking and unblocking threads.
        self.cond = Condition()

    def wait(self):
        """
        Main entry point for threads to wait at the barrier.

        A thread calling this method will block until `num_threads`
        threads have also called `wait()`, after which all waiting
        threads are released.
        """
        self.cond.acquire() # Acquires the lock associated with the condition variable.
        self.count_threads -= 1 # Decrements the count of threads yet to reach the barrier.

        # Block Logic: Checks if this is the last thread to reach the barrier.
        if self.count_threads == 0:
            self.cond.notify_all() # Notifies all waiting threads to resume execution.
            # Resets the counter for the next use of the barrier.
            self.count_threads = self.num_threads
        else:
            self.cond.wait() # Waits (blocks) until notified by the last thread to reach the barrier.

        self.cond.release() # Releases the lock after proceeding.

