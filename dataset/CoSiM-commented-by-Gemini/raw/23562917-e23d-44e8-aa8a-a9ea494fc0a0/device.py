


"""
This module defines a distributed device simulation framework that executes scripts
in parallel threads and uses a semaphore-based barrier for synchronization.

It features a `Device` class representing a network node, a `DeviceThread` that
manages the device's lifecycle, and a `ScriptThread` for parallel script execution.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

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
        self.barrier = None
        self.devices = []
        self.locks = {}
        self.lock_used = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared synchronization barrier for all devices.

        Args:
            devices (list): A list of all devices in the network.
        """
        
        
        barrier = ReusableBarrierSem(len(devices))

        
        for device in devices:
            self.devices.append(device)
            device.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device and ensures a lock exists for the script's
        location.

        Args:
            script (Script): The script object to execute.
            location (int): The location identifier associated with the script's data.
        """
        
        if script is not None:
            self.scripts.append((script, location))
            
            for device in self.devices:
                if device.locks.get(location) is not None:
                    self.locks[location] = device.locks[location]
                    self.lock_used = 1
                    break

            
            if self.lock_used is None:
                self.locks[location] = Lock()

            self.lock_used = None
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

class ScriptThread(Thread):
    """
    A thread dedicated to executing a single script.
    """

    def __init__(self, device, script, location, neighbours):
        """
        Initializes the script execution thread.

        Args:
            device (Device): The parent device.
            script (Script): The script to execute.
            location (int): The data location for the script.
            neighbours (list): A list of neighboring devices.
        """
        
        Thread.__init__(self)


        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Acquires a lock, gathers data, executes the script, and propagates the
        result back to the relevant devices before releasing the lock.
        """
        
        with self.device.locks[self.location]:
            script_data = []
            
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = self.script.run(script_data)

                


                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)


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
        self.script_threads = []

    def run(self):
        """
        The main execution loop for the device.

        This loop waits for a timepoint, then creates and starts a `ScriptThread`
        for each assigned script. After all script threads have completed, it
        synchronizes with other devices at a global barrier.
        """
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                thread = ScriptThread(self.device, script, location, neighbours)
                self.script_threads.append(thread)

            for thread in self.script_threads:
                thread.start()
            for thread in self.script_threads:
                thread.join()
            
            self.script_threads = []

            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
