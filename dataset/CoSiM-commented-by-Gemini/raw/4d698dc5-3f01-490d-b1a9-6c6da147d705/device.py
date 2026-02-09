"""
@file device.py
@brief Implements a device model for a distributed simulation with lazy lock initialization.

This file defines a `Device` class and its `DeviceThread` for a sensor network
simulation. It features a unique mechanism for lazy initialization of location-based
locks, where a lock for a specific location is created and distributed to all
devices only when the first script for that location is assigned.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    """
    Represents a device in the simulation.

    Each device manages its own sensor data and executes scripts. It uses a shared
    barrier for timepoint synchronization and a lazily initialized set of locks
    to protect data access on a per-location basis.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary of the device's sensor readings.
            supervisor: The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # Pre-allocates a list to hold location-specific locks.
        self.locks = [None] * 100
        self.devices = None
        self.barrier = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for all devices.

        If the barrier has not been set, this method initializes a reusable
        semaphore-based barrier and distributes its reference to all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices
        # Block Logic: The first device to call this will create the shared barrier.
        if self.barrier is None:
            self.barrier = ReusableBarrierSem(len(devices))
            # Invariant: Ensure all devices share the exact same barrier instance.
            for i in self.devices:
                i.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be run and handles lazy initialization of locks.

        When a script for a new location is assigned, this method creates a lock
        for that location and distributes it to all other devices to ensure
        they share the same lock instance.

        Args:
            script: The script object to execute.
            location (int): The location index for the script and its lock.
        """
        if script is not None:
            # Block Logic: Lazily initialize the lock for a given location.
            # If the lock doesn't exist, create it and propagate it to all other devices.
            if self.locks[location] is None:
                self.locks[location] = Lock()
                for i in self.devices:
                    i.locks[location] = self.locks[location]

            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device.

    This thread orchestrates the device's participation in the simulation,
    including script execution and timepoint synchronization.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device thread."""
        while True:
            # Block Logic: Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value signals the end of the simulation.
                break

            # Waits for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()

            
            # Block Logic: Execute all assigned scripts for the current timepoint.
            for (script, location) in self.device.scripts:
                script_data = []

                # Pre-condition: Acquire the lock for the specific location to ensure
                # that data gathering and updates are atomic for that location.
                self.device.locks[location].acquire()
                
                # Block Logic: Gather data from neighbors and the local device.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Invariant: script_data now holds all relevant data for the location.
                if script_data != []:
                    
                    result = script.run(script_data)

                    # Block Logic: Propagate the script result back to the local
                    # device and all its neighbors.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                
                # Release the lock after the operation is complete.
                self.device.locks[location].release()

            # Clear the event for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Wait at the barrier for all other devices to finish the timepoint.
            self.device.barrier.wait()