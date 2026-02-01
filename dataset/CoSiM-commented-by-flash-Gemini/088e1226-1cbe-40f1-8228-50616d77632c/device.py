"""
This module implements a distributed device simulation framework, focusing on concurrent script
execution and robust synchronization using a semaphore-based reusable barrier.

Algorithm:
- Device: Represents a simulated physical device with sensor data and script processing capabilities.
- MyThread: A worker thread responsible for executing a single script on collected data and updating
  device states, ensuring data consistency with location-specific locks.
- DeviceThread: The orchestrating thread for each `Device`, which distributes scripts to `MyThread`
  instances and manages overall synchronization using shared locks and barriers.
- ReusableBarrierSem: A semaphore-based synchronization primitive ensuring all participating threads
  reach a specific point before proceeding.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
class Device(object):
    """
    Represents a simulated device in a distributed environment.

    Each device has a unique ID, manages its sensor data, and can receive
    and execute scripts. It uses shared locks and a reusable barrier for
    thread-safe operations and synchronization with other devices.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings,
                            where keys represent locations.
        supervisor (Supervisor): A reference to the central supervisor managing devices.
        script_received (threading.Event): Event to signal when new scripts are assigned.
        scripts (list): A list to store assigned scripts (script, location) tuples.
        timepoint_done (threading.Event): Event to signal completion of a timepoint's tasks.
                                          (Not explicitly used in run() but present).
        devices (list): A list of all Device objects in the simulation. (Set by `setup_devices`).
        barrier (ReusableBarrierSem): A shared semaphore-based reusable barrier for synchronization.
        thread (DeviceThread): The dedicated orchestrating thread for this device.
        locations (list): A list of `threading.Lock` objects, one for each possible data location,
                          to provide fine-grained access control.
        data_lock (threading.Lock): A general lock for protecting modifications to `sensor_data`.
        get_lock (threading.Lock): A general lock for protecting reads from `sensor_data`.
        setup (threading.Event): Event to signal that shared resources have been set up.
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
        # Event for signaling when new scripts have been received.
        self.script_received = Event()
        # List to hold (script, location) tuples assigned to this device.
        self.scripts = []
        # Event for signaling that the processing for a timepoint is done (currently unused).
        self.timepoint_done = Event()
        self.devices = None # Will be set by setup_devices.
        self.barrier = None # Will be set by setup_devices.
        # Create and start a dedicated thread for this device's operations.
        self.thread = DeviceThread(self)
        self.locations = [] # List of location-specific locks, initialized by setup_devices.
        self.data_lock = Lock() # Lock for protecting set_data operations.
        self.get_lock = Lock() # Lock for protecting get_data operations.
        self.setup = Event() # Event to signal that shared resources are ready.
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
        Initializes shared synchronization primitives (barrier and location-specific locks)
        across all devices. This method ensures these resources are only initialized once
        by the device with `device_id == 0`.

        Pre-condition: This method should be called by a central entity (e.g., supervisor)
                       after all devices are instantiated.
        Invariant: If `device_id` is 0, a new `ReusableBarrierSem` is created, and a list of
                   `threading.Lock` objects (for 100 locations) is initialized. These shared
                   resources are then propagated to all `Device` instances, and their `setup`
                   event is set.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Store a reference to all devices.
        self.devices = devices
        barrier = ReusableBarrierSem(len(devices))
        # Block Logic: Only the device with ID 0 initializes shared resources.
        if self.device_id == 0:
            # Initialize 100 location-specific locks.
            for _ in range(100):
                self.locations.append(Lock())

            # Propagate the created barrier and location locks to all devices,
            # and signal that setup is complete.
            for dev in devices:
                dev.barrier = barrier
                dev.locations = self.locations
                dev.setup.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location for this device.
        If `script` is None, it signals that script assignments for the current timepoint are complete.

        Args:
            script (Script or None): The script object to execute.
            location (int): The data location where the script should be applied.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that script assignment for the current timepoint is complete.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.
        Acquires a general read lock to protect access to sensor data.

        Args:
            location (int): The location from which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location
                 does not exist in the device's sensor_data.
        """
        # Acquire a general lock to protect reading from sensor_data.
        with self.get_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.
        Acquires a general write lock to protect modification of sensor data.

        Args:
            location (int): The location at which to set data.
            data (any): The new data value to set.
        """
        # Acquire a general lock to protect writing to sensor_data.
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Joins the device's dedicated thread, effectively waiting for it to complete
        its execution before the program exits.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The dedicated orchestrating thread for a `Device` object.

    This thread is responsible for:
    1. Interacting with the supervisor to get neighbor information.
    2. Waiting for shared resources to be set up.
    3. Waiting for script assignments.
    4. Spawning `MyThread` instances to execute scripts concurrently.
    5. Managing the execution of `MyThread` instances in controlled batches.
    6. Synchronizing with other `DeviceThread` instances using a shared barrier.

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

        Invariant: The loop continues until the supervisor signals termination.
                   Within each iteration, it waits for setup completion, retrieves
                   neighbor information, waits for new script assignments,
                   creates and manages a pool of `MyThread` workers to execute
                   these scripts in batches, and then synchronizes with other
                   `DeviceThread` instances via a barrier.
        """
        # Wait until shared resources (barrier, locations locks) are set up by device 0.
        self.device.setup.wait()
        while True:
            threads = [] # List to hold MyThread instances.
            # Retrieve information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: If no neighbors are returned (e.g., simulation termination signal), break the loop.
            if neighbours is None:
                break
            # Wait for the `script_received` event, which signals that script assignments
            # for the current timepoint are complete and ready for processing.
            self.device.script_received.wait()
            # Clear the `script_received` event, resetting it for the next timepoint.
            self.device.script_received.clear()
            i = 0
            # Block Logic: Create MyThread instances for each assigned script.
            for _ in self.device.scripts:
                threads.append(MyThread(self.device, self.device.scripts, neighbours, i))
                i = i + 1
            scripts_rem = len(self.device.scripts)
            start = 0
            # Functional Utility: Handle concurrent execution of MyThreads.
            # If fewer than 8 scripts, start all at once.
            if len(self.device.scripts) < 8:
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            # Block Logic: If 8 or more scripts, process them in batches of 8 (or remaining scripts).
            else:
                while True:
                    if scripts_rem == 0:
                        break
                    # If more than 8 scripts remaining, process a batch of 8.
                    if scripts_rem >= 8:
                        for i in xrange(start, start + 8): # xrange for Python 2 compatibility.
                            threads[i].start()
                        for i in xrange(start, start + 8): # xrange for Python 2 compatibility.
                            threads[i].join()
                        start = start + 8
                        scripts_rem = scripts_rem - 8
                    # If less than 8 scripts remaining, process the rest.
                    else:
                        for i in xrange(start, start + scripts_rem): # xrange for Python 2 compatibility.
                            threads[i].start()
                        for i in xrange(start, start + scripts_rem): # xrange for Python 2 compatibility.
                            threads[i].join()
                        break
            # Clear the list of scripts after processing.
            self.device.scripts = []
            # Synchronize with other DeviceThreads using the shared barrier.
            self.device.barrier.wait()

class MyThread(Thread):
    """
    A worker thread responsible for executing a single script.

    It acquires a location-specific lock, gathers data from the device and its neighbors,
    executes the script, updates the data, and then releases the lock.

    Attributes:
        device (Device): The Device object associated with this worker thread.
        scripts (list): A reference to the list of scripts in the parent DeviceThread.
        neighbours (list): A list of neighboring Device objects.
        indice (int): The index of the script in `self.scripts` that this thread will execute.
    """

    def __init__(self, device, scripts, neighbours, indice):
        """
        Initializes a new MyThread instance.

        Args:
            device (Device): The Device associated with this worker.
            scripts (list): The list of scripts from the DeviceThread.
            neighbours (list): List of neighboring Devices.
            indice (int): The index of the script to execute from the scripts list.
        """
        Thread.__init__(self, name="My Worker Thread for Device %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours
        self.indice = indice

    def run(self):
        """
        The main execution method for the MyThread.

        Invariant: Acquires a location-specific lock, collects data from the device
                   and its neighbors, executes the script, updates data in both
                   the current and neighboring devices, and then releases the lock.
        """
        (script, location) = self.scripts[self.indice]
        # Acquire a location-specific lock to ensure exclusive access for data at this location.
        self.device.locations[location].acquire()
        script_data = []
        # Block Logic: Collect data from neighboring devices for the current location.
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        # Collect data from the current device for the current location.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        # Block Logic: If data is available, execute the script and update results.
        if script_data != []:
            # Execute the script with the collected data.
            result = script.run(script_data)
            # Update the sensor data in neighboring devices.
            for device in self.neighbours:
                device.set_data(location, result)
            # Update the sensor data in the current device.
            self.device.set_data(location, result)
        self.device.locations[location].release() # Release the location-specific lock.
