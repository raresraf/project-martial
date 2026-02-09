

"""
This module defines a distributed device simulation framework where each script
is executed in a separate thread.

It features a `Device` class representing a network node, a `DeviceThread` that
manages the device's lifecycle, and a `RunScripts` class that defines the execution
logic for a single script. Synchronization is handled by an external reusable
barrier and per-location locks.
"""

from threading import Event, Thread, Lock
import my_barrier
import my_thread

class Device(object):
    """
    Represents a device in a distributed network that can process sensor data.

    Each device runs a main thread (`DeviceThread`) which in turn spawns
    `RunScripts` threads for each assigned script. It uses a shared barrier
    for synchronization and locks for data access.
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
        self.threads = []
        self.barrier = None
        self.lock = [None] * 50

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the list of devices and the shared synchronization barrier.

        Args:
            devices (list): A list of all devices in the network.
        """
        
        
        
        for device in devices:
            self.devices.append(device)
        
        self.barrier = my_barrier.ReusableBarrierCond(len(devices))
        for device in devices:
            device.barrier = self.barrier

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
            self.script_received.set()
            
            for device in self.devices:
                if self.lock[location] is None and device.lock[location] is not None:
                    self.lock[location] = device.lock[location]
            if self.lock[location] is None:
                self.lock[location] = Lock()
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

        This loop waits for a timepoint to be triggered, then creates and starts
        a `RunScripts` thread for each assigned script. It waits for all script
        threads to complete before synchronizing with other devices at a global
        barrier.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break



            self.device.timepoint_done.wait()
            
            for (script, location) in self.device.scripts:
            	
                thread = my_thread.RunScripts(self.device, neighbours, location, script)
                self.device.threads.append(thread)
            for i in xrange(len(self.device.threads)):
                self.device.threads[i].start()
            for i in xrange(len(self.device.threads)):


                self.device.threads[i].join()

            self.device.threads = []
            self.device.barrier.wait()
            self.device.timepoint_done.clear()


from threading import Thread

class RunScripts(Thread):
    """
    A thread dedicated to executing a single script.
    """
    
    def __init__(self, device, neighbours, location, script):
        """
        Initializes the script execution thread.

        Args:
            device (Device): The parent device.
            neighbours (list): A list of neighboring devices.
            location (int): The data location for the script.
            script (Script): The script to execute.
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.location = location
        self.script = script

    def run(self):
        """
        Acquires a lock for the script's location, gathers data, executes the
        script, and propagates the result back to the relevant devices before
        releasing the lock.
        """
        self.device.lock[self.location].acquire()
        self.device.script_received.wait()
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
        self.device.lock[self.location].release()

