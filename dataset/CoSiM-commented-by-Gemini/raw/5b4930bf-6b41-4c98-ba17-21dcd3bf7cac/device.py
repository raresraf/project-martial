"""
@file device.py
@brief Defines a device model for a distributed simulation.

This file contains the `Device` and `DeviceThread` classes. The `DeviceThread`
is designed to spawn worker threads to execute scripts.

@note This code appears to be incomplete as it references a `MyThread` class
      that is not defined within this file. The documentation assumes `MyThread`
      is a `threading.Thread` subclass for script execution.
"""

from threading import Event, Thread
from barrier import *

class Device(object):
    """
    Represents a single device in the simulation, managing sensor data and
    coordinating script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
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
        self.devices = []
        self.barrier = None
        self.threads = []

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared barrier and records the list of all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: The first device to enter this method creates a new
        # shared barrier and distributes it to all other devices.
        if self.barrier is None:
            barrier = ReusableBarrierSem(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
        
        # This loop populates the local list of all devices.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script: The script object to run.
            location: The location context for the script.
        """
        flag = 0 # This flag is unused.
        if script is not None:
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
    The main control thread for a Device.

    It orchestrates the execution of scripts for each timepoint by spawning
    worker threads and synchronizing with a shared barrier.
    """

    def __init__(self, device):
        """
        Initializes the main device thread.
        Args:
            device (Device): The parent device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop for the device.

        @note This method relies on a `MyThread` class which is not defined in this file.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals simulation end.
                break
            
            # Waits for the supervisor to assign all scripts for the timepoint.
            self.device.timepoint_done.wait()

            # Block Logic: For each assigned script, create a worker thread.
            # Assumes `MyThread` is a Thread subclass that takes the device,
            # location, script, and neighbors as arguments.
            for (script, location) in self.device.scripts:
                mythread = MyThread(self.device, location, script, neighbours)
                self.device.threads.append(mythread)

            # Block Logic: Start all worker threads, then wait for them all to complete.
            for xthread in self.device.threads:
                xthread.start()
            for xthread in self.device.threads:
                xthread.join()

            # Clean up thread list for the next timepoint.
            self.device.threads = []
            
            self.device.timepoint_done.clear()
            # Wait at the barrier for all other devices to finish the timepoint.
            self.device.barrier.wait()