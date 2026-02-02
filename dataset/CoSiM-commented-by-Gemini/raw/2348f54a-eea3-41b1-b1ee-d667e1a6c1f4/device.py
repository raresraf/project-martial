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
        self.scripts_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.data_lock = Lock()
        self.list_locks = {}
        self.barrier = None
        self.devices = None

    def __str__(self):
        """
        Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's connection to other devices in the network,
        including a shared barrier for synchronization.

        Args:
            devices: A list of all devices in the network.
        """
        self.devices = devices

        # The first device in the list is responsible for creating the
        # synchronization barrier and locks for all devices.
        if self.device_id == self.devices[0].device_id:
            self.barrier = barrier.ReusableBarrierCond(len(self.devices))
            for dev in self.devices:
                for location in dev.sensor_data:
                    self.list_locks[location] = Lock()
        else:
            # Other devices get the barrier and locks from the first device.
            self.barrier = devices[0].get_barrier()
            self.list_locks = devices[0].get_list_locks()
        
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device. If the script is None,
        it signals that all scripts have been received.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set()


    def get_barrier(self):
        """
        Returns the synchronization barrier used by the devices.
        """
        return self.barrier

    def get_list_locks(self):
        """
        Returns the dictionary of locks for sensor data locations.
        """
        return self.list_locks

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location, with thread safety.
        """
        with self.data_lock:
            if location in self.sensor_data:
                data = self.sensor_data[location]
            else:
                data = None
        return data

    def set_data(self, location, data):
        """
        Updates the sensor data for a specific location, with thread safety.
        """
        with self.data_lock:
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

    def run(self):
        """
        The main loop of the device thread. It continuously waits for scripts,
        executes them in parallel, and then waits at a barrier for all
        other devices to complete their execution for the current step.
        """
        while True:
            # Get the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait until all scripts for the current step are received.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            # Execute the assigned scripts in parallel using ScriptThread.
            threads = []
            for (script, location) in self.device.scripts:
                threads.append(
                    ScriptThread(self.device, script, location, neighbours))
                if len(threads) == 8:
                    for thr in threads:
                        thr.start()
                    for thr in threads:
                        thr.join()
                    threads = []
            
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()

            # Wait at the barrier for all devices to finish the current step.
            self.device.barrier.wait()



class ScriptThread(Thread):
    """
    A thread for executing a single script on a device's sensor data.
    """

    def __init__(self, device, script, location, neighbours):
        """
        Initializes the script thread.

        Args:
            device: The device on which the script will be executed.
            script: The script object to be executed.
            location: The location of the sensor data to be processed.
            neighbours: A list of neighboring devices.
        """
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script. It acquires a lock for the specified location,
        gathers data from the device and its neighbors, runs the script, and
        then updates the data on all involved devices with the result.
        """
        
        # Acquire a lock to ensure exclusive access to the data at this location.
        self.device.list_locks[self.location].acquire()

        script_data = []

        # Collect data from neighboring devices.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Collect data from the current device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Run the script if there is any data to process.
        if script_data != []:
            result = self.script.run(script_data)

            # Update the data on all neighboring devices and the current device.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        # Release the lock.
        self.device.list_locks[self.location].release()