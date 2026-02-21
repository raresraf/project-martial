


from threading import Thread, Lock
from barrier import ReusableBarrier


class Device(object):
    """
    Represents a simulated device in a distributed system, capable of managing
    sensor data, executing assigned scripts, and coordinating with other devices
    through a supervisor and shared synchronization primitives.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding the device's sensor data,
                                keyed by location.
            supervisor (Supervisor): The central supervisor managing the devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # List to store (script, location) tuples assigned to this device.
        self.scripts = []
        # The main thread pool controlling this device's operations.
        self.thread = DeviceThreadPool(self)
        
        # Global barrier shared among all devices, set by the leader device.
        self.barrier = None
        
        # Internal barrier used for synchronizing script assignment completion within this device.
        self.inner_barrier = ReusableBarrier(2) # Expects two participants: the assigner and the DeviceThreadPool.
        
        # Global lock shared among all devices, primarily for supervisor interaction.
        self.lock = None
        
        # Internal lock for protecting this device's sensor data during updates.
        self.inner_lock = Lock()
        
        # Global map of locks, providing fine-grained locking for each data location.
        self.lock_map = None

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization primitives (barrier, global lock, and lock map)
        among a collection of devices. This method is designed to be called once
        by a supervisor or an orchestrating entity. The device with the minimum
        ID acts as the initializer for these shared resources.

        Args:
            devices (list): A list of all Device objects participating in the simulation.
        """
        # Collect all device IDs to determine the leader.
        device_ids = [device.device_id for device in devices]
        # The device with the minimum ID is designated as the leader.
        leader_id = min(device_ids)

        # Only the leader device initializes and distributes the shared resources.
        if self.device_id == leader_id:
            # Create a reusable barrier for all devices.
            barrier = ReusableBarrier(len(devices))
            # Create a global lock for operations requiring exclusive access across devices.
            lock = Lock()
            # Create a map to hold per-location locks for fine-grained data access control.
            lock_map = {}
            # Distribute the shared synchronization primitives to all devices.
            for device in devices:
                device.set_barrier(barrier)
                device.set_lock(lock)
                device.set_lock_map(lock_map)
                # Start the DeviceThreadPool for each device after setup.
                device.thread.start()

    def set_barrier(self, barrier):
        """
        Sets the global shared barrier for this device.

        Args:
            barrier (ReusableBarrier): The shared barrier instance.
        """
        self.barrier = barrier

    def set_lock(self, lock):
        """
        Sets the global shared lock for this device.

        Args:
            lock (Lock): The shared global lock instance.
        """
        self.lock = lock

    def set_lock_map(self, lock_map):
        """
        Sets the global shared lock map for this device.

        Args:
            lock_map (dict): The shared dictionary of per-location locks.
        """
        self.lock_map = lock_map

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a given data location.
        If the location is new, it ensures a lock is present in the global lock_map.
        If `script` is None, it signals the completion of script assignments for a timepoint.

        Args:
            script (Script): The script object to be executed.
            location: The data location pertinent to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Acquire the global lock to safely update the lock_map.
            with self.lock:
                # Initialize a new lock for this location if it doesn't exist.
                if location not in self.lock_map:
                    self.lock_map[location] = Lock()
        else:
            # Signal completion of script assignments for this timepoint by waiting on inner barrier.
            # This allows the DeviceThreadPool to proceed with script execution.
            self.inner_barrier.wait()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location.

        Args:
            location: The key for the sensor data.

        Returns:
            The sensor data for the specified location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Sets or updates the sensor data for a specific location.
        Uses an internal lock to protect against concurrent modifications.

        Args:
            location: The key for the sensor data to be updated.
            data: The new data value.
        """
        if location in self.sensor_data:
            with self.inner_lock: # Protects this device's sensor_data dictionary.
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device by joining its associated thread pool.
        """
        self.thread.join()


class DeviceThreadPool(Thread):
    """
    Manages the concurrent execution of scripts for a single Device.
    It orchestrates the time-stepped simulation, including supervisor interaction,
    script dispatching to worker threads, and synchronization with other devices.
    """
    def __init__(self, device):
        """
        Initializes the DeviceThreadPool.

        Args:
            device (Device): The Device object that this thread pool manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device's thread pool.
        It manages simulation timepoints, fetches neighbors, dispatches
        scripts to worker threads, and synchronizes with a global barrier.
        """
        while True:
            # Acquire the global lock before interacting with the supervisor to get neighbors.
            with self.device.lock:
                neighbours = self.device.supervisor.get_neighbours()

            # If no neighbors are returned, it signifies the end of the simulation.
            if neighbours is None:
                break

            # Wait on the internal barrier, which is released by assign_script(None),
            # signaling that all scripts for the current timepoint have been assigned.
            self.device.inner_barrier.wait()

            threads = [] # List to hold worker threads.

            # Create and start a worker DeviceThread for each assigned script.
            for (script, location) in self.device.scripts:
                thread = DeviceThread(self.device, script, location, neighbours)
                thread.start()
                threads.append(thread)

            # Wait for all worker threads to complete their script execution.
            for thread in threads:
                thread.join()

            # Clear the list of scripts as they have all been processed.
            self.device.scripts = [] 

            # Synchronize with all other devices at the global barrier before starting the next timepoint.
            self.device.barrier.wait()


class DeviceThread(Thread):
    """
    A worker thread responsible for executing a single script within a DeviceThreadPool.
    It handles data collection from local and neighboring devices, applies a script,
    and propagates the results, ensuring thread-safe data access using per-location locks.
    """
    def __init__(self, device, script, location, neighbours):
        """
        Initializes a DeviceThread worker.

        Args:
            device (Device): The local device object.
            script (Script): The script to be executed.
            location: The data location associated with the script.
            neighbours (list): A list of neighboring Device objects to interact with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        The execution logic for the worker thread. It acquires a lock for its
        specific data location, collects relevant data, runs the assigned script,
        and then updates local and neighboring device data.
        """
        # Acquire the per-location lock to ensure exclusive access to data at this location.
        with self.device.lock_map[self.location]:
            script_data = []
            # Collect data from neighboring devices for the current location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            # Collect data from the local device for the current location.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # If data was collected, execute the script.
            if len(script_data) != 0:
                # Execute the script with the aggregated data.
                result = self.script.run(script_data)

                # Propagate the script's result back to all neighboring devices.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                # Update the local device's data with the script's result.
                self.device.set_data(self.location, result)
