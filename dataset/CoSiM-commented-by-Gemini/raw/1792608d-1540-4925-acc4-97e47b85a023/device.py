


"""
This module defines a distributed device simulation framework that executes scripts
in parallel threads and uses a barrier for synchronization.

It features a `Device` class representing a network node, a `DeviceThread` that
manages the device's lifecycle, a `MyThread` class for script execution, and a
`ReusableBarrier` for synchronization.
"""

from threading import Lock, Thread, Event, Condition

max_size = 100

class MyThread(Thread):
    """
    A thread dedicated to executing a single script on a device and its neighbors.
    """
    
    def __init__(self, dev_thread, neighbors, location, script):
        """
        Initializes the script execution thread.

        Args:
            dev_thread (DeviceThread): The parent device thread.
            neighbors (list): A list of neighboring devices.
            location (int): The data location for the script.
            script (Script): The script to execute.
        """
        Thread.__init__(self)
        self.dev_thread = dev_thread
        self.neighbors = neighbors


        self.location = location
        self.script = script

    def run(self):
        """
        Acquires a lock, gathers data, executes the script, and propagates the
        result back to the relevant devices before releasing the lock.
        """
        self.dev_thread.device.location_lock[self.location].acquire()
        script_data = []
        
        for device in self.neighbors:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.dev_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            
            for device in self.neighbors:
                device.set_data(self.location, result)
            
            self.dev_thread.device.set_data(self.location, result)
        self.dev_thread.device.location_lock[self.location].release()

class ReusableBarrier():
    """
    A reusable barrier for synchronizing a fixed number of threads.
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
        self.cond_barrier = None
        self.location_lock = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared synchronization objects for all devices.
        Only the first device in the list is responsible for creating these objects.

        Args:
            devices (list): A list of all devices in the network.
        """

        if self.device_id == devices[0].device_id:
            self.cond_barrier = ReusableBarrier(len(devices))
            self.location_lock = []
            for i in range(0, max_size):
                self.location_lock.append(Lock())
            for dev in devices:
                dev.cond_barrier = self.cond_barrier
                dev.location_lock = self.location_lock


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

        This loop waits for scripts to be received, then creates and starts a
        `MyThread` for each script. After all script threads have completed,
        it synchronizes with other devices at a global barrier.
        """
        
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()


            self.device.script_received.clear()

            thread_list = []
            for (script, location) in self.device.scripts:
                thread_list.append(MyThread(self, neighbours, location, script))

            for thr in thread_list:
                thr.start()

            for thr in thread_list:
                thr.join()

            
            self.device.cond_barrier.wait()
            self.device.timepoint_done.wait()
