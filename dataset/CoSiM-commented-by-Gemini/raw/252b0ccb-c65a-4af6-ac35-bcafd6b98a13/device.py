


"""
This module defines a distributed device simulation framework that executes scripts
in parallel threads.

NOTE: This file is incomplete as it is missing the definition for the `MyThread`
class, which is used for script execution.

It features a `Device` class representing a network node and a `DeviceThread` that
manages the device's lifecycle and the creation of `MyThread` instances.
"""

from threading import Event, Thread
from barrier import *

class Device(object):
    """
    Represents a device in a distributed network that can process sensor data.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data, keyed by location.
            supervisor (Supervisor): A supervisor object that manages the network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()


        self.thread = DeviceThread(self)
        self.thread.start()
        self.devices = []
        self.barrier = None
        self.threads = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared synchronization barrier for all devices.

        Args:
            devices (list): A list of all devices in the network.
        """
        

        if self.barrier is None:


            barrier = ReusableBarrierSem(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
        for device in devices:
            if device is not None:
                self.devices.append(device)
    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        Args:
            script (Script): The script object to execute.
            location (int): The location identifier associated with the script's data.
        """
        
        flag = 0
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location to retrieve data for.

        Returns:
            The sensor data, or None if the location is not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location.

        Args:
            location (int): The location to update data for.
            data: The new data value.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread."""
        
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device instance.
    """

    def __init__(self, device):
        """
        Initializes the device thread.

        Args:
            device (Device): The device that this thread will run.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device.

        This loop waits for a timepoint, then creates and starts a `MyThread`
        for each assigned script. After all script threads have completed, it
        synchronizes with other devices at a global barrier.
        """

        while True:
            

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                mythread = MyThread(self.device, location, script, neighbours)
                self.device.threads.append(mythread)

            for xthread in self.device.threads:
                xthread.start()
            for xthread in self.device.threads:
                xthread.join()

            
            self.device.threads = []
            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
