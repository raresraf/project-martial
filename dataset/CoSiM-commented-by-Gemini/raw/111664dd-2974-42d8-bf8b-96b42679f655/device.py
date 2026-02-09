

"""
This module defines a distributed device simulation framework that uses a thread pool
for concurrent script execution.

It features a `Device` class representing a network node, a `Helper` class to manage
script execution in a thread pool, and a `DeviceThread` for the main device loop.
Synchronization between devices is achieved using a reusable barrier and per-location
locks for data access.
"""

from threading import Event, Thread, Lock
from multiprocessing.dummy import Pool as ThreadPool
from reusablebarrier import ReusableBarrierCond

class Device(object):
    """
    Represents a device in a distributed network that can process sensor data.

    Each device runs in its own thread and uses a thread pool to execute multiple
    scripts concurrently. It synchronizes with other devices using a shared reusable
    barrier and uses locks to protect access to its sensor data.
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
        self.barrier = None
        self.data_locks = {}
        for location in sensor_data:
            self.data_locks[location] = Lock()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's view of the network, including the synchronization barrier.
        Only the device with ID 0 is responsible for creating and distributing the barrier.

        Args:
            devices (list): A list of all devices in the network.
        """
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script (Script): The script object to execute.
            location (int): The location identifier associated with the script's data.
        """
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location, acquiring a lock to ensure safety.

        Args:
            location (int): The location to retrieve data for.

        Returns:
            The sensor data, or None if the location is not found.
        """
        
        if location in self.sensor_data:
            self.data_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location and releases the lock.

        Args:
            location (int): The location to update data for.
            data: The new data value.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_locks[location].release()

    def shutdown(self):
        """Shuts down the device's thread."""
        
        self.thread.join()

class Helper(object):
    """
    A helper class that manages script execution using a thread pool.
    """
    
    def __init__(self, device):
        """
        Initializes the Helper with a reference to the parent device and a thread pool.

        Args:
            device (Device): The parent device.
        """
        
        self.device = device
        self.pool = ThreadPool(8)
        self.neighbours = None
        self.scripts = None

    def set_neighbours_and_scripts(self, neighbours, scripts):
        """
        Sets the context for script execution.

        Args:
            neighbours (list): The list of neighboring devices.
            scripts (list): The list of scripts to execute.
        """
        
        self.neighbours = neighbours
        self.scripts = scripts

    def script_run(self, (script, location)):
        """
        The target function for the thread pool. It executes a single script.

        Args:
            (script, location) (tuple): A tuple containing the script and its location.
        """
        
        script_data = []
        for device in self.neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        if script_data != []:
            result = script.run(script_data)
            for device in self.neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)
            self.device.set_data(location, result)

    def run(self):
        """
        Asynchronously executes all assigned scripts in the thread pool.
        """
        
        self.pool.map_async(self.script_run, self.scripts)

    def close_pool(self):
        """
        Closes the thread pool and waits for all tasks to complete.
        """
        
        self.pool.close()
        self.pool.join()


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
        self.helper = None

    def run(self):
        """
        The main execution loop for the device.

        This loop waits for scripts or a timepoint completion signal, then uses a
        `Helper` instance to execute the scripts in a thread pool. After script
        execution is complete, it synchronizes with other devices at a global barrier.
        """

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.helper = Helper(self.device)
            


            while True:
                if (self.device.script_received.is_set() or
                self.device.timepoint_done.is_set()):
                    
                    
                    
                    if self.device.script_received.is_set():
                        self.device.script_received.clear()
                        self.helper.set_neighbours_and_scripts(neighbours,
							self.device.scripts)
                        self.helper.run()
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break
            self.helper.close_pool()
            self.device.barrier.wait()
