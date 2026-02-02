"""
This module defines the Device class and its associated threads for simulating a
network of devices that process sensor data.
"""

from threading import Event, Thread, Lock
import barrier

class Device(object):
    """
    Represents a single device in the simulated network. Each device has a
    unique ID, holds sensor data, and is managed by a supervisor. It can
    execute scripts on its data and synchronize with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: A dictionary containing the device's sensor data,
                         keyed by location.
            supervisor: The supervisor object that manages this device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.locks = None
        self.barrier = None

    def __str__(self):
        """
        Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's connection to other devices in the network,
        including a shared barrier and locks for synchronization.

        Args:
            devices: A list of all devices in the network.
        """
        devices[0].barrier = barrier.ReusableBarrierSem(len(devices))
        devices[0].locks = {}
        list_index = list(range(len(devices)))
        for i in list_index[1:len(devices)]:
            devices[i].barrier = devices[0].barrier
            devices[i].locks = devices[0].locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device. If the script is None,
        it signals that all scripts for the current timepoint have been received.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specific location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device's thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread of execution for a device. This thread waits for scripts,
    executes them, and synchronizes with other devices.
    """

    def __init__(self, device):
        """
        Initializes the device thread.

        Args:
            device: The Device object that this thread belongs to.
        """


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def thread_script(self, neighbours, script, location):
        """
        Executes a single script. It acquires a lock for the specified location,
        gathers data from the device and its neighbors, runs the script, and
        then updates the data on all involved devices with the result.

        Args:
            neighbours: A list of neighboring devices.
            script: The script object to be executed.
            location: The location of the sensor data to be processed.
        """

        
        script_data = []
        if location not in self.device.locks:
            self.device.locks[location] = Lock()

        


        self.device.locks[location].acquire()

        
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data)

            
            


            for device in neighbours:
                device.set_data(location, result)

            
            self.device.set_data(location, result)

        
        self.device.locks[location].release()

    def run(self):
        """
        The main loop of the device thread. It continuously waits for a timepoint
        to be completed, executes all assigned scripts in parallel, and then
        waits at a barrier for all other devices to finish.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            threads_script = []
            for (script, location) in self.device.scripts:
                
                thread = Thread(target=self.thread_script,
                    args=(neighbours, script, location))
                thread.start()
                threads_script.append(thread)

            
            for j in xrange(len(threads_script)):
                threads_script[j].join()

            
            self.device.barrier.wait()
            self.device.timepoint_done.clear()