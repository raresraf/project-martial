


"""
This module defines a distributed device simulation framework that uses a manually
implemented thread pool to execute scripts in parallel.

It features a `Device` class representing a network node, a `DeviceThread` that
manages the device's lifecycle and the thread pool, and a `runScripts` function
that serves as the target for the execution threads.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

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
        self.thread = DeviceThread(self)
        self.thread.start()
        self.dictLocks = {}

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
            self.barrier = ReusableBarrierCond(len(devices))

            for device in devices:
                for location in device.sensor_data.keys():
                    if self.dictLocks.has_key(location) == False:
                        self.dictLocks[location] = Lock()
                device.setup_mutualBarrier(self.barrier, self.dictLocks)


    def setup_mutualBarrier(self, barrier, dictLocks):
        """
        Assigns the shared barrier and lock dictionary to this device.

        Args:
            barrier (ReusableBarrierCond): The shared barrier instance.
            dictLocks (dict): The shared dictionary of locks.
        """
        
        if self.device_id != 0:
            self.barrier = barrier
            self.dictLocks = dictLocks


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


def runScripts((script, location), neighbours, callingDevice):
    """
    The target function for script execution threads. It acquires a lock for the
    script's location, gathers data, executes the script, and propagates the result.

    Args:
        (script, location) (tuple): A tuple containing the script and its location.
        neighbours (list): A list of neighboring devices.
        callingDevice (Device): The device that initiated the script execution.
    """

    script_data = []
    
    callingDevice.dictLocks[location].acquire()
    for device in neighbours:
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)
            
    data = callingDevice.get_data(location)
    if data is not None:
        script_data.append(data)

    if script_data != []:
        
        result = script.run(script_data)

        


        for device in neighbours:
            device.set_data(location, result)
            
            callingDevice.set_data(location, result)
    callingDevice.dictLocks[location].release()



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

        This loop waits for scripts to be received, then manually creates a pool
        of threads in batches of 8 to execute the scripts in parallel. After all
        scripts are executed, it synchronizes with other devices at a global barrier.
        """
        
        
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()

            threadsList = []
            index = 0
            nrScripts = len(self.device.scripts)
            
            while nrScripts:
                
                if nrScripts > 7:
                    for j in range(8):
                        threadsList.append(
                        Thread(target=runScripts, args=
                        (self.device.scripts[index], neighbours, self.device)))
                        index += 1
                    nrScripts = nrScripts - 8
                else:
                    for j in range(nrScripts):
                        threadsList.append(
                        Thread(target=runScripts, args=
                        (self.device.scripts[index], neighbours, self.device)))
                        index += 1
                    nrScripts = 0

                
                for j in range(len(threadsList)):
                    threadsList[j].start()

                
                for j in range(len(threadsList)):
                    threadsList[j].join()

                threadsList = []

            
            self.device.script_received.clear()
            
            self.device.barrier.wait()
