from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier


class Device(object):
    """
    Represents a device in a simulated distributed system.
    Each device utilizes a pool of worker threads to execute scripts concurrently,
    synchronizing through global and internal barriers, and managing data access
    via per-location locks.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding the device's sensor data,
                                keyed by location.
            supervisor (Supervisor): The supervisor object responsible for
                                     managing devices and providing neighborhood information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # List of 8 Events, one for each worker thread, to signal script availability.
        self.script_received = []
        for _ in xrange(8):
            self.script_received.append(Event())
        # List to store assigned scripts.
        self.scripts = []
        # Reference to all devices in the simulation, set during setup_devices.
        self.devices = None
        # Global barrier for synchronizing all devices. Set by setup_devices.
        self.barrier = None
        # Global lock for coordinating access to shared resources like the 'locks' dictionary.
        self.lock = Lock()
        # Dictionary of Locks, providing fine-grained locking for per-location data access.
        self.locks = {}
        # Stores a reference to the current neighbors, updated by DeviceThread with thread_id 0.
        self.neighbours = None

        # Internal barrier to synchronize the 8 worker threads within this device.
        thread_barrier = ReusableBarrier(8)
        self.threads = []
        # Create and start 8 worker DeviceThread instances for this device.
        for i in xrange(8):
            self.threads.append(DeviceThread(self, i, thread_barrier))
        for i in xrange(8):
            self.threads[i].start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures shared resources among devices. All devices get a reference
        to the list of all devices. The device with ID 0 initializes the global
        barrier and a global lock for managing the `locks` dictionary.

        Args:
            devices (list): A list of all Device objects participating in the simulation.
        """
        self.devices = devices # Store the list of all devices for later access.
        # Only the device with device_id == 0 performs the initial setup of global shared resources.
        if self.device_id == 0:
            # Initialize the global barrier for all 8 * len(devices) worker threads.
            barrier = ReusableBarrier(8 * len(devices))
            # Initialize a global lock for managing the 'locks' dictionary.
            lock = Lock()
            # Distribute the global barrier and lock to all devices.
            for device in devices:
                device.barrier = barrier
                device.lock = lock

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.
        Handles initialization of per-location locks if a new location is encountered.
        If `script` is None, it signals the completion of script assignments for the timepoint.

        Args:
            script (Script): The script object to be executed.
            location: The data location pertinent to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            
            # If a new location is encountered, create a new Lock for it and
            # distribute it to all devices' 'locks' dictionary.
            if location not in self.locks:
                with self.lock: # Acquire global lock to protect 'locks' dictionary modification.
                    auxlock = Lock()
                    for device in self.devices:
                        device.locks[location] = auxlock
        else:
            # Signal to all 8 worker threads that scripts are ready for processing.
            for i in xrange(8):
                self.script_received[i].set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location: The key for the sensor data.

        Returns:
            The sensor data for the specified location, or None if not found.
        """
        # Data access is expected to be protected externally by acquiring a lock from self.locks[location].
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location: The key for the sensor data.
            data: The new data to set.
        """
        # Data access is expected to be protected externally by acquiring a lock from self.locks[location].
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining all its worker threads.
        """
        # Signal shutdown to worker threads (e.g., by sending None script or using a stop flag)
        # and then join them.
        for i in xrange(8):
            # To gracefully shut down, each script_received event would need to be set
            # so the worker unblocks, checks for a stop signal (not present in current code),
            # and then exits. For now, simply joining them, assuming an external stop mechanism.
            self.threads[i].join()


class DeviceThread(Thread):
    """
    Represents a single worker thread within a Device. Each Device has 8 such
    threads that concurrently execute assigned scripts for a timepoint.
    """
    def __init__(self, device, thread_id, barrier):
        """
        Initializes a DeviceThread worker.

        Args:
            device (Device): The Device object to which this worker belongs.
            thread_id (int): A unique identifier for this worker thread (0-7).
            barrier (ReusableBarrier): An internal barrier for synchronizing the 8 workers within the device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id
        self.barrier = barrier

    def run(self):
        """
        The main execution loop for the worker thread. It handles supervisor interaction
        (for thread_id 0), internal synchronization, script processing, and global
        barrier synchronization before proceeding to the next timepoint.
        """
        while True:
            # Only thread_id 0 retrieves neighbor information from the supervisor.
            # This result is then used by all other worker threads in the same device.
            # WARNING: This relies on timely and consistent update of self.device.neighbours
            # across all workers after this call and before they proceed.
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # Synchronize all 8 worker threads within this device.
            self.barrier.wait()
            # If no neighbors, it signifies the end of the simulation for all workers.
            if self.device.neighbours is None:
                break

            # Wait for this specific worker's script_received event to be set,
            # indicating that scripts are ready for its assigned tasks.
            self.device.script_received[self.thread_id].wait()
            
            # Distribute scripts among the 8 worker threads in a round-robin fashion.
            # Each worker processes a subset of the assigned scripts.
            # WARNING: The 'self.device.scripts' list is not cleared after processing.
            # This means the same scripts will be executed repeatedly in subsequent timepoints,
            # which is likely an unintended behavior in a time-stepped simulation.
            for i in xrange(self.thread_id, len(self.device.scripts), 8):
                (script, location) = self.device.scripts[i]
                # Acquire the per-location lock to ensure exclusive access to data at this location.
                with self.device.locks[location]:
                    script_data = []
                    
                    # Collect data from neighboring devices for the current location.
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    # Collect data from the local device for the current location.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    # Execute the script if there is data to process.
                    if script_data != []:
                        # Execute the script with the aggregated data.
                        result = script.run(script_data)

                        # Propagate the script's result back to all neighboring devices.
                        for device in self.device.neighbours:
                            device.set_data(location, result)
                        
                        # Update the local device's data with the script's result.
                        self.device.set_data(location, result)

            # Clear this worker's script_received event for the next cycle.
            self.device.script_received[self.thread_id].clear()
            # It should clear the main device's script list here, which is a bug in the original code.
            # self.device.scripts = []

            # Synchronize all devices (all 8 * len(devices) worker threads) at the global barrier
            # before starting the next timepoint.
            self.device.barrier.wait()