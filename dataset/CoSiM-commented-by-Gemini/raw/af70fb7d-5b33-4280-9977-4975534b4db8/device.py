
"""
This module implements a multi-threaded simulation of a distributed system of devices.

This version uses a custom `ReusableBarrier` for synchronization and creates a new
thread for each script to be executed.
"""

from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This barrier implementation uses a two-phase protocol to ensure that all
    threads wait at the barrier before any of them are allowed to proceed.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have arrived.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Implements one phase of the barrier.

        Args:
            count_threads (list[int]): A list containing the count of remaining threads.
            threads_sem (Semaphore): The semaphore to signal when all threads have arrived.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a device in the distributed system.
    
    Each device has an ID, sensor data, and can execute scripts. It runs in its
    own thread and communicates with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data for the device.
            supervisor (Supervisor): The supervisor object that manages the device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barrier = None
        self.list_thread = []
        self.thread.start()
        self.location_lock = [None] * 100

    def __str__(self):
        """
        Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device with a list of other devices in the system.

        Initializes and shares a single barrier instance among all devices.

        Args:
            devices (list[Device]): A list of all devices in the system.
        """
        
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        for device in devices:
            if device is not None:
                self.devices.append(device)


    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        This method also handles the complex logic of sharing locks for specific
        locations among devices.

        Args:
            script: The script to be executed.
            location: The location associated with the script.
        """
        flag = 0
        if script is not None:
            self.scripts.append((script, location))
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        flag = 1
                        break
                if flag == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.

        Args:
            location: The location from which to retrieve data.

        Returns:
            The sensor data at the given location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location.

        Args:
            location: The location to update.
            data: The new data to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its thread.
        """
        self.thread.join()


class MyThread(Thread):
    """
    A thread for executing a single script on a device.
    """
    def __init__(self, device, location, script, neighbours):
        """
        Initializes the MyThread.

        Args:
            device (Device): The device executing the script.
            location: The location associated with the script.
            script: The script to execute.
            neighbours (list[Device]): A list of neighboring devices.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script.

        It acquires a lock for the script's location, gathers data from
        neighboring devices, runs the script, and then updates the data on all neighbors.
        """
        self.device.location_lock[self.location].acquire()
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
        self.device.location_lock[self.location].release()


class DeviceThread(Thread):
    """
    The main thread of execution for a Device.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread.

        It waits for a timepoint to end, then creates and starts a new thread
        for each assigned script. After all script threads have completed, it
        synchronizes with other devices using a barrier.
        """
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            for thread_elem in self.device.list_thread:
                thread_elem.start()
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            self.device.list_thread = []

            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
