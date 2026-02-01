"""
This module implements a simulation framework for distributed devices.
It includes classes for representing devices, managing their execution in separate threads,
and executing scripts on sensor data. It relies on a `ReusableBarrier` for synchronization.

Algorithm:
- Device: Manages sensor data, receives scripts, and coordinates with a supervisor.
- DeviceThread: Orchestrates the execution of assigned scripts by spawning `ScriptThread` workers.
- ScriptThread: Executes individual scripts on sensor data, ensuring data consistency with locks.
"""

from threading import Event, Thread, Lock, Semaphore
from ReusableBarrier import ReusableBarrier


class Device(object):
    """
    Represents a simulated device in a distributed environment.

    Each device has a unique ID, manages its own sensor data, and interacts
    with a supervisor. It can receive scripts for execution, and it uses
    synchronization primitives (barrier, locks, semaphores) to coordinate
    its operations with other devices.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings,
                            where keys represent locations.
        supervisor (Supervisor): A reference to the central supervisor managing devices.
        script_received (threading.Event): Event to signal when a script has been assigned.
        scripts (list): A list to store assigned scripts (script, location) tuples.
        lock_setter (threading.Lock): Lock to protect `set_data` operations.
        lock_getter (threading.Lock): Lock to protect `get_data` operations.
        lock_assign (threading.Lock): Lock to protect `assign_script` operations.
        barrier (ReusableBarrier): A shared barrier for synchronizing all devices.
        location_lock (dict): A dictionary of `threading.Lock` objects, one for each
                              data location, to ensure exclusive access during data manipulation.
        semaphore (threading.Semaphore): A semaphore to limit the number of concurrently
                                         executing `ScriptThread` instances.
        thread (DeviceThread): The dedicated thread for this device's operations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): Initial sensor data for the device.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal the receipt of new scripts.
        self.script_received = Event()
        # List to hold (script, location) tuples assigned to this device.
        self.scripts = []

        # Locks to protect critical sections during data access and script assignment.
        self.lock_setter = Lock()
        self.lock_getter = Lock()
        self.lock_assign = Lock()

        # Shared synchronization primitives, initialized by `setup_devices`.
        self.barrier = None
        self.location_lock = {}

        # Semaphore to control the number of concurrent script executions.
        self.semaphore = Semaphore(8)

        # Create and start a dedicated thread for this device's operations.
        self.thread = DeviceThread(self)

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives (barrier, location-specific locks)
        across all devices. This method is typically called once by a coordinating entity
        (e.g., the supervisor) for the device with `device_id == 0`.

        Pre-condition: This method should ideally be called only once by a designated device
                       (e.g., device_id 0) to set up shared resources.
        Invariant: If `device_id` is 0, a new `ReusableBarrier` is created, and
                   `location_lock` objects are instantiated for all unique sensor data
                   locations across all devices. These shared resources are then
                   propagated to all other `Device` instances.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Only the device with ID 0 is responsible for initializing
        # shared synchronization primitives to avoid redundant creation.
        if self.device_id == 0:
            # Initialize a reusable barrier for all devices.
            self.barrier = ReusableBarrier(len(devices))

            # Block Logic: Create a lock for each unique sensor data location across all devices.
            for device in devices[:]:
                for loc in device.sensor_data.keys():
                    if loc not in self.location_lock:
                        self.location_lock[loc] = Lock()

            # Block Logic: Propagate the shared barrier and location locks to all devices,
            # and start their respective DeviceThread instances.
            for device in devices[:]:
                device.barrier = self.barrier
                device.location_lock = self.location_lock
                # Start the dedicated thread for each device.
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location for this device.
        If `script` is None, it signals that no more scripts are coming for the current
        timepoint, and sets the `script_received` event.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The data location where the script should be applied.
        """
        # Ensure thread-safe assignment of scripts.
        with self.lock_assign:
            if script is not None:
                # Add the script and its target location to the device's script queue.
                self.scripts.append((script, location))
            else:
                # If script is None, it indicates the end of script assignment for a timepoint.
                self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (int): The location from which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location
                 does not exist in the device's sensor_data.
        """
        # Ensure thread-safe retrieval of data.
        with self.lock_getter:
            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (int): The location at which to set data.
            data (any): The new data value to set.
        """
        # Ensure thread-safe modification of data.
        with self.lock_setter:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Joins the device's dedicated thread, effectively waiting for it to complete
        its execution before the program exits.
        """
        self.thread.join()


class ScriptThread(Thread):
    """
    A worker thread responsible for executing a single assigned script on sensor data
    for a specific device and its neighbors. It uses location-specific locks and a
    semaphore for managing concurrency and data consistency.

    Attributes:
        script (Script): The script object to execute.
        device_thread (DeviceThread): The parent `DeviceThread` instance.
        location (int): The data location relevant to this script execution.
        neighbours (list): A list of neighboring `Device` objects from which to fetch data.
    """

    def __init__(self, device_thread, script, location, neighbours):
        """
        Initializes a new ScriptThread instance.

        Args:
            device_thread (DeviceThread): The parent DeviceThread that spawned this worker.
            script (Script): The script to be executed.
            location (int): The data location the script will operate on.
            neighbours (list): A list of neighboring devices to interact with.
        """
        # Initialize the base Thread class.
        Thread.__init__(self)
        self.script = script
        self.device_thread = device_thread
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        The main execution method for the ScriptThread.

        Invariant: Acquires a location-specific lock and a concurrency semaphore
                   before accessing and modifying sensor data. Collects data
                   from the associated device and its neighbors, executes the
                   script, updates data, and then releases the locks and semaphore.
        """
        # Acquire a lock for the specific data location to ensure exclusive access
        # for operations on this location across all devices.
        self.device_thread.device.location_lock[self.location].acquire()

        # Acquire a semaphore permit to control the number of concurrently executing
        # ScriptThreads for this device, preventing resource exhaustion.
        self.device_thread.device.semaphore.acquire()

        script_data = []

        # Block Logic: Collect data from neighboring devices for the current location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Collect data from the current device for the current location.
        data = self.device_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: If there is data collected, execute the script and update results.
        if script_data != []:
            # Execute the script with the collected data.
            result = self.script.run(script_data)

            # Update the sensor data in neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            # Update the sensor data in the current device.
            self.device_thread.device.set_data(self.location, result)

        # Release the semaphore permit.
        self.device_thread.device.semaphore.release()
        # Release the lock for the specific data location.
        self.device_thread.device.location_lock[self.location].release()


class DeviceThread(Thread):
    """
    Manages the lifecycle and script execution for a single Device.

    This thread continuously checks for new scripts, spawns `ScriptThread` workers
    to process them, and synchronizes with other device threads using a barrier.
    It acts as the main processing loop for a device in the simulation.

    Attributes:
        device (Device): The Device object associated with this thread.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        # Initialize the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.

        Invariant: The loop continues until the supervisor signals termination
                   by returning None for neighbors.
                   Within each iteration, the thread waits for `script_received` signal,
                   processes assigned scripts using `ScriptThread` instances,
                   and then synchronizes with other devices via a barrier before clearing
                   its script reception signal.
        """
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors are returned (e.g., simulation termination signal), break the loop.
            if neighbours is None:
                break

            # Wait for the `script_received` event, which signals that scripts have been assigned
            # for the current timepoint and are ready for processing.
            self.device.script_received.wait()
            script_threads = []

            # Block Logic: Iterate through each assigned script and create a dedicated
            # ScriptThread to execute it in parallel.
            for (script, location) in self.device.scripts:
                # Create a new ScriptThread for each script.
                thread = ScriptThread(self, script, location, neighbours)
                script_threads.append(thread)
                # Start the ScriptThread to begin execution.
                thread.start()

            # Block Logic: Wait for all ScriptThreads to complete their execution.
            for thread in script_threads:
                thread.join()

            # Clear the `script_received` event, indicating that all scripts for this
            # timepoint have been processed.
            self.device.script_received.clear()

            # Synchronize with other DeviceThreads using the shared barrier, ensuring
            # all devices complete their timepoint processing before proceeding.
            self.device.barrier.wait()
