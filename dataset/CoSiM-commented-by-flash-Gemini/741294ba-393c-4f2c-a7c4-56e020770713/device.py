"""
This module implements a multi-threaded device simulation framework.

It defines `Device` objects that represent simulated entities, a `DeviceThread`
to manage the main operational loop of each device, and `Worker` threads
that execute specific data processing scripts. The framework supports
inter-device communication and script processing in a concurrent manner,
with explicit per-location locking for data consistency and a shared,
semaphore-based reusable barrier (`ReusableBarrierSem` from `barrier.py`)
for global synchronization.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem # Custom barrier class for global synchronization.


class Device(object):
    """
    Represents a single simulated device in the system.

    Each device manages its own `sensor_data`, interacts with a `supervisor`,
    and processes scripts. It coordinates with other devices through a shared
    barrier and manages access to data locations using shared per-location locks.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping locations (int) to sensor data values.
            supervisor (object): A reference to the supervisor object managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data # The local sensor data for this device.

        self.supervisor = supervisor # Reference to the supervisor managing all devices.

        # `script_received` Event is used to signal that new scripts have been assigned.
        self.script_received = Event()

        # `scripts` is a list of (script_object, location) tuples assigned to this device.
        self.scripts = []

        # `lock_locations` will be a shared list of Lock objects, one for each data location.
        # It's populated by device 0 and then shared among all devices.
        self.lock_locations = []

        # `barrier` is a ReusableBarrierSem instance for global synchronization.
        # It is initialized with a placeholder value (0) and will be properly set up by device 0.
        self.barrier = ReusableBarrierSem(0)

        # `thread` is the main DeviceThread responsible for this device's control flow.
        self.thread = DeviceThread(self)

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global shared synchronization primitives (`barrier` and `lock_locations`).
        This method is designed to be called once by all devices, but global initialization
        logic is handled by the device with `device_id` 0. It also starts the `DeviceThread`.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Global initialization for `barrier` and `lock_locations`:
        # Only device with `device_id` 0 performs this setup and distributes the resources.
        if self.device_id == 0:
            # Create the global shared barrier with the total number of devices.
            barrier = ReusableBarrierSem(len(devices))

            # Determine the maximum data location ID to correctly size the `lock_locations` list.
            nr_locations = 0
            for i in range(len(devices)):
                for location in devices[i].sensor_data.keys():
                    if location > nr_locations:
                        nr_locations = location
            nr_locations += 1 # Ensure size is sufficient for max_location_id + 1.

            # Initialize a global list of Locks, one for each possible data location.
            for i in range(nr_locations):
                lock_location = Lock()
                self.lock_locations.append(lock_location)

            # Distribute the initialized barrier and shared `lock_locations` list to all devices.
            for i in range(len(devices)):
                devices[i].barrier = barrier # Assign the shared barrier.
                # Copy references to the shared `lock_locations` list for each device.
                for j in range(nr_locations):
                    devices[i].lock_locations.append(self.lock_locations[j])
                
                # Start the DeviceThread for each device.
                devices[i].thread.start()
        # For non-device 0, `self.barrier` and `self.lock_locations` will be populated by device 0.

    def assign_script(self, script, location):
        """
        Assigns a processing script and its associated data location to this device.
        If `script` is None, it signals that script assignment for the current
        timepoint is complete.

        Args:
            script (object): The script object (must have a `run` method) to be executed,
                             or None to signal timepoint completion.
            location (int): The data location this script pertains to.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script to the device's list.
        else:
            # If `script` is None, it signals the end of script assignments for this timepoint.
            self.script_received.set() # Signal that all scripts have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location within this device.

        Note: This method itself does *not* acquire any locks to protect `self.sensor_data`.
        The caller (`Worker.solve_script`) is responsible for acquiring the appropriate
        per-location lock from `self.lock_locations` before calling this method
        to ensure thread-safe access to `self.sensor_data`.

        Args:
            location (int): The location ID for which to retrieve data.

        Returns:
            Any: The data at the specified location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location within this device.

        Note: Similar to `get_data`, this method itself does *not* acquire any locks
        to protect `self.sensor_data` modification. The caller (`Worker.solve_script`)
        is responsible for acquiring the appropriate per-location lock from
        `self.lock_locations` before calling this method.

        Args:
            location (int): The location ID for which to set data.
            data (Any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the graceful shutdown sequence for the device's main thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a `Device`.

    It is responsible for interacting with the supervisor to get neighbor
    information, managing timepoint progression, and dynamically spawning
    `Worker` threads to execute scripts. It coordinates global synchronization
    using a shared barrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously processes timepoints:
        1. Retrieves neighbor information from the supervisor.
        2. Waits for scripts to be assigned.
        3. Spawns `Worker` threads for each assigned script.
        4. Waits for all `Worker` threads to complete.
        5. Participates in global barrier synchronization.
        """
        workers = [] # List to hold `Worker` thread instances for the current timepoint.

        while True:
            # Retrieves the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # If no neighbors are returned (e.g., shutdown signal from supervisor),
            # break the loop and terminate this thread.
            if neighbours is None:
                break

            # Waits until new scripts have been assigned to this device for the current timepoint.
            # This ensures that `self.device.scripts` is fully populated for the timepoint.
            self.device.script_received.wait()
            self.device.script_received.clear() # Clear the event for the next script assignment cycle.

            # Create a `Worker` thread for each assigned script.
            for (script, location) in self.device.scripts:
                workers.append(Worker(self.device, script,
                                        location, neighbours))

            # Start all `Worker` threads.
            for i in range(len(workers)):
                workers[i].start()

            # Wait for all `Worker` threads to complete their assigned tasks.
            for i in range(len(workers)):
                workers[i].join()

            # Clear the list of workers for the next timepoint.
            workers = []

            # Participates in the global barrier synchronization, waiting for all devices
            # to complete their current timepoint processing of scripts.
            self.device.barrier.wait()


class Worker(Thread):
    """
    A worker thread responsible for executing a single assigned script task.

    Instances of `Worker` are created and managed by `DeviceThread`. It handles
    collecting data, executing the script, and updating data in relevant devices,
    all while explicitly managing per-location synchronization.
    """
    
    def __init__(self, device, script, location, neighbours):
        """
        Initializes a Worker instance.

        Args:
            device (Device): The Device instance this worker operates for.
            script (object): The script object (must have a `run` method) to be executed.
            location (int): The data location relevant to this script.
            neighbours (list): The list of neighboring devices for the current timepoint.
        """
        Thread.__init__(self, name="Worker for Device %d, Location %d" % (device.device_id, location))
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def solve_script(self, script, location, neighbours):
        """
        Executes a given script, collecting data from neighboring devices and
        the local device, and then distributing the results. This method
        explicitly manages the acquisition and release of the per-location lock.

        Args:
            script (object): The script object to execute.
            location (int): The data location for the script.
            neighbours (list): List of neighboring devices.
        """
        # Acquire the global lock for the specific data location to ensure exclusive access.
        # This prevents race conditions when multiple workers/devices access the same location concurrently.
        self.device.lock_locations[location].acquire()

        script_data = [] # List to collect all relevant data for the script.

        # Gathers data from all neighboring devices for the current location.
        # `device.get_data` relies on this external lock for thread safety.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Gathers data from its own device for the current location.
        # `self.device.get_data` relies on this external lock for thread safety.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # If any data was collected, run the script and update devices.
        if script_data != []:
            result = script.run(script_data) # Execute the script.

            # Updates the data in neighboring devices with the script's result.
            # `device.set_data` relies on this external lock for thread safety.
            for device in neighbours:
                device.set_data(location, result)

            # Updates its own device's data with the script's result.
            # `self.device.set_data` relies on this external lock for thread safety.
            self.device.set_data(location, result)

        # Releases the global lock for the data location.
        self.device.lock_locations[location].release()

    def run(self):
        """
        The main execution logic for `Worker`.
        It simply calls `solve_script` to process its assigned task.
        """
        self.solve_script(self.script, self.location, self.neighbours)