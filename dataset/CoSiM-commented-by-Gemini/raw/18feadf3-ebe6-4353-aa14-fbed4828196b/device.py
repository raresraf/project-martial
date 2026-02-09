


"""
This module defines a distributed device simulation framework where scripts are
executed in parallel threads.

It features a `Device` class representing a network node, a `DeviceThread` that
manages the device's lifecycle, an `ExecuteScripts` class for running scripts
in separate threads, and a `ReusableBarrierCond` for synchronization.
"""

from threading import Event, Thread, Lock, Condition
from execute_scripts import ExecuteScripts

class ReusableBarrierCond(object):
    """
    A reusable barrier for synchronizing a fixed number of threads using a condition variable.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads to synchronize.
        """

        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
    def wait(self):
        """
        Causes the calling thread to wait until all threads have reached the barrier.
        """
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


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
        self.locks = {}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier and locks for all devices.
        Only the device with ID 0 is responsible for creating these objects.

        Args:
            devices (list): A list of all devices in the network.
        """
        
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            locks = {}
            for device in devices:
                device.barrier = barrier
                device.locks = locks
        for location in self.sensor_data.keys():
            if not location in self.locks:
                self.locks[location] = Lock()

    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        Args:
            script (Script): The script object to execute.
            location (int): The location identifier associated with the script's data.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location to retrieve data for.

        Returns:
            The sensor data, or None if the location is not found.
        """
        
        loc = location
        return self.sensor_data[loc] if location in self.sensor_data else None

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

        This loop waits for scripts to be received, then creates and starts an
        `ExecuteScripts` thread for each script. After all script threads have
        completed, it synchronizes with other devices at a global barrier.
        """
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()
            self.device.script_received.clear()
            
            tlist = []
            for (script, location) in self.device.scripts:
                
                thread = ExecuteScripts(self.device, location, \
                        neighbours, script)
                tlist.append(thread)
                thread.start()
            for thread in tlist:
                thread.join()
            
            
            self.device.barrier.wait()

from threading import Thread

class ExecuteScripts(Thread):
    """
    A thread dedicated to executing a single script.
    """
    
    def __init__(self, device, location, neighbours, script):
        """
        Initializes the script execution thread.

        Args:
            device (Device): The parent device.
            location (int): The data location for the script.
            neighbours (list): A list of neighboring devices.
            script (Script): The script to execute.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def run(self):
        """
        Acquires a lock, gathers data, executes the script, and propagates the
        result back to the relevant devices before releasing the lock.
        """
        
        script_data = []
        self.device.locks[self.location].acquire()
        


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
        self.device.locks[self.location].release()
