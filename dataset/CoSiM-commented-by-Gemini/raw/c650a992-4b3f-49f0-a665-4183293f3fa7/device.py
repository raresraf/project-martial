"""
This module defines a simulated device for a distributed sensing network.

It provides classes for a `Device`, a `DeviceThreadPool` for managing its
main lifecycle, and `DeviceThread` for executing specific scripts. The simulation
uses threading primitives to coordinate actions between multiple devices, such
as data aggregation and updates.
"""

from threading import Thread, Lock
from barrier import ReusableBarrier


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device has a unique ID, local sensor data, and can execute scripts.
    It synchronizes with other devices using shared barriers and locks, managed
    by a supervisor entity.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor data.
            supervisor (object): The supervisor object that manages the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.thread = DeviceThreadPool(self)
        
        # Shared barrier for all devices in the network
        self.barrier = None
        
        # Barrier for internal synchronization within the device's threads
        self.inner_barrier = ReusableBarrier(2)
        
        # Shared lock for device-level operations
        self.lock = None
        
        # Lock for protecting the device's internal state
        self.inner_lock = Lock()
        
        # A map of locks for specific data locations
        self.lock_map = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up a group of devices with shared synchronization primitives.

        The device with the lowest ID is designated as the leader and is
        responsible for creating the shared barrier and locks.

        Args:
            devices (list): A list of Device objects to be set up.
        """
        
        device_ids = [device.device_id for device in devices]
        leader_id = min(device_ids)

        # The leader device initializes the shared synchronization objects.
        if self.device_id == leader_id:
            barrier = ReusableBarrier(len(devices))
            lock = Lock()
            lock_map = {}
            for device in devices:
                device.set_barrier(barrier)
                device.set_lock(lock)
                device.set_lock_map(lock_map)
                device.thread.start()

    def set_barrier(self, barrier):
        """Assigns the shared barrier to the device."""
        self.barrier = barrier

    def set_lock(self, lock):
        """Assigns the shared lock to the device."""
        self.lock = lock

    def set_lock_map(self, lock_map):
        """Assigns the shared lock map to the device."""
        self.lock_map = lock_map

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script (object): The script object to be executed.
            location (str): The data location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))

            with self.lock:
                # Create a new lock for this location if one doesn't already exist.
                if location not in self.lock_map:
                    self.lock_map[location] = Lock()
        else:
            # If no script is assigned, wait on the inner barrier.
            self.inner_barrier.wait()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.

        Args:
            location (str): The data location to retrieve from.

        Returns:
            The sensor data at the given location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location in a thread-safe manner.

        Args:
            location (str): The data location to update.
            data: The new data to be stored.
        """
        if location in self.sensor_data:
            with self.inner_lock:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThreadPool(Thread):
    """Manages the main execution lifecycle of a device."""

    def __init__(self, device):
        """
        Initializes the device's thread pool.

        Args:
            device (Device): The device this thread pool belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the device."""
        while True:
            # Acquire the device lock to get the list of neighbors.
            with self.device.lock:
                neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break

            # Synchronize with the script assignment phase.
            self.device.inner_barrier.wait()

            threads = []

            # Create and start a new thread for each assigned script.
            for (script, location) in self.device.scripts:
                thread = DeviceThread(self.device, script, location, neighbours)
                thread.start()
                threads.append(thread)

            # Wait for all script threads to complete.
            for thread in threads:
                thread.join()

            # Wait for all devices to finish their execution cycle.
            self.device.barrier.wait()


class DeviceThread(Thread):
    """A thread dedicated to executing a single script on a device."""

    def __init__(self, device, script, location, neighbours):
        """
        Initializes a script execution thread.

        Args:
            device (Device): The device executing the script.
            script (object): The script to be executed.
            location (str): The data location for the script.
            neighbours (list): A list of neighboring devices.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """Executes the script."""

        # Acquire the lock for the specific data location.
        with self.device.lock_map[self.location]:

            script_data = []
            # Gather data from all neighboring devices.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Include the device's own data.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if len(script_data) != 0:
                # Run the script with the aggregated data.
                result = self.script.run(script_data)

                # Propagate the result to all neighbors and the device itself.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)
