"""
A simulation framework for a network of communicating devices.

This module provides classes to simulate a network of devices that execute
scripts and share data. It includes a custom `ReusableBarrier` implementation
using semaphores, a `Device` class, and a multi-threaded execution model where a
`DeviceThread` spawns multiple `MyThread`s to execute scripts.
"""


from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    A reusable two-phase barrier implemented with semaphores.

    This barrier ensures that all participating threads synchronize at two points
    before proceeding.
    """
    
    def __init__(self, num_threads):
        """Initializes the Barrier with the number of threads to wait for."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Waits for all threads to complete both phases of the barrier."""
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """A single phase of the barrier synchronization."""
        
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a single device in the simulated network.

    Each device has a set of scripts to execute, sensor data, and can
    communicate with its neighbors. The device's logic is driven by a
    DeviceThread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: A dictionary of sensor data for this device.
            supervisor: A supervisor object that manages the network.
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
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up static resources for all devices.

        This method is intended to be called to initialize shared resources for
        all devices in the simulation, such as a shared barrier and a list of
        all other devices.

        Args:
            devices: A list of all devices in the simulation.
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
        Assigns a script to be executed by this device.

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
        Retrieves sensor data for a given location.

        Args:
            location: The location to get data from.

        Returns:
            The sensor data for the given location, or None if not available.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets the sensor data for a given location.

        Args:
            location: The location to set data for.
            data: The new data value.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        
        self.thread.join()


class MyThread(Thread):
    """
    A worker thread that executes a single script for a device.
    """
    
    def __init__(self, device, location, script, neighbours):
        """
        Initializes a MyThread instance.

        Args:
            device: The parent device.
            location: The location associated with the script.
            script: The script to execute.
            neighbours: A list of neighboring devices.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        The main execution logic for the worker thread.

        This thread acquires a lock for the script's location, gathers data from
        neighbors, runs the script, and then disseminates the result.
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
    The main thread for a device, responsible for orchestrating worker threads.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The Device object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device thread.

        This loop waits for a timepoint to be triggered, then spawns multiple
        MyThread workers to execute the device's scripts in parallel. After the
        workers complete, it synchronizes with other devices using a barrier.
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
