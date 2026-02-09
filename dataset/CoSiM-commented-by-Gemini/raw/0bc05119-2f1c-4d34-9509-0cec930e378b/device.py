

"""
This module defines a distributed device simulation framework with enhanced synchronization.

It includes a `Device` class representing a network node, a `Barrier` class for thread
synchronization, and a `DeviceThread` for concurrent execution. This version introduces
per-location locking to ensure data consistency during script execution, in addition to
a global barrier for timepoint synchronization.
"""

from threading import Event, Thread, Lock, Condition

class Barrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This class provides a synchronization point where threads can wait until all
    participating threads have reached the barrier. The number of threads to wait for
    is specified at initialization.
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

        When a thread calls wait(), it decrements a counter. If the counter reaches
        zero, all waiting threads are notified. Otherwise, the thread blocks until
        it is notified.
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

    Each device runs in its own thread and can execute scripts on data collected from
    itself and its neighbors. It uses a shared barrier for synchronizing timepoints
    and per-location locks for mutually exclusive access to sensor data.
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
        self.locks = [None] * 100
        self.devices = []

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's view of the network, including the synchronization barrier.

        Args:
            devices (list): A list of all devices in the network.
        """
        
        
        self.devices = devices
        if self.barrier is None:
            self.barrier = Barrier(len(devices))
            self.assign_barrier(devices, self.barrier)

    def assign_barrier(self, devices, barrier):
        """
        Assigns a shared barrier to all devices in the network.

        Args:
            devices (list): The list of all devices.
            barrier (Barrier): The shared barrier instance.
        """

        for device in devices:
            if device.device_id != 0:
                device.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device and sets up a lock for its location.

        Args:
            script (Script): The script object to execute.
            location (int): The location identifier associated with the script's data.
        """


        if script is not None:
            if self.locks[location] is None:
                self.locks[location] = Lock()
                for device in self.devices:
                    device.locks[location] = self.locks[location]
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
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

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
    The execution thread for a Device instance.

    This thread contains the main loop where the device processes scripts and
    synchronizes with other devices.
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

        The loop consists of the following steps:
        1. Wait for the current timepoint to complete for all devices.
        2. Acquire a lock for the specific data location being processed.
        3. Execute the script on data from this device and its neighbors.
        4. Update the data on this device and its neighbors with the script's result.
        5. Release the lock.
        6. Wait at a global barrier to ensure all devices have finished the timepoint.
        """
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                self.device.locks[location].acquire()
                script_data = []
                
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

            self.device.barrier.wait()

            
            self.device.timepoint_done.clear()
