"""
A simulation framework for a network of communicating devices.

This module provides classes to simulate a network of devices that execute
scripts and share data. It includes a custom two-phase barrier implemented with
semaphores (`ReusableBarrierSem`), a `Device` class, and a multi-threaded
execution model where a `DeviceThread` spawns multiple `MiniDeviceThread`s to
execute scripts.
"""


from threading import Thread, Semaphore, Event, Lock

class ReusableBarrierSem(object):
    """
    A reusable two-phase barrier implemented with semaphores.

    This barrier ensures that all participating threads synchronize at two points
    before proceeding.
    """
    
    def __init__(self, num_threads):
        """Initializes the Barrier with the number of threads to wait for."""
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)
    def wait(self):
        """Waits for all threads to complete both phases of the barrier."""
        
        self.phase1()
        self.phase2()
    def phase1(self):
        """The first phase of the barrier synchronization."""
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):


                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    def phase2(self):
        """The second phase of the barrier synchronization."""
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

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
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.locks = []
        
        self.nrlocks = max(sensor_data)
    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up static resources for all devices.

        This method is intended to be called by one device to initialize
        shared resources for all devices in the simulation, such as a shared
        barrier and locks for locations.

        Args:
            devices: A list of all devices in the simulation.
        """
        
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            
            for _, device in enumerate(devices):
                device.barrier = self.barrier
        
        if self.device_id == 0:
            listmaxim = []
            for _, device in enumerate(devices):
                listmaxim.append(device.nrlocks)
            
            number = max(listmaxim)
            
            for _ in range(number + 1):
                self.locks.append(Lock())
            
            for _, device in enumerate(devices):
                device.locks = self.locks
    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device.

        Args:
            script: The script to be executed.
            location: The location associated with the script.
        """
        
        if script is not None:
            self.scripts.append((script, location))
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

        if location in self.sensor_data:
            data = self.sensor_data[location]
        else: data = None
        return data
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

class MiniDeviceThread(Thread):
    """
    A worker thread that executes a single script for a device.
    """
    
    def __init__(self, device, script, location, neighbours):
        """
        Initializes a MiniDeviceThread.

        Args:
            device: The parent device.
            script: The script to execute.
            location: The location associated with the script.
            neighbours: A list of neighboring devices.
        """
        
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        The main execution logic for the worker thread.

        This thread acquires a lock for the script's location, gathers data from
        neighbors, runs the script, and then disseminates the result.
        """
	
        self.device.locks[self.location].acquire()
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
        
        self.device.locks[self.location].release()


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
        self.nr_iter = None

    def run(self):
        """
        The main execution loop for the device thread.

        This loop waits for a timepoint to be triggered, then spawns multiple
        MiniDeviceThread workers to execute the device's scripts in parallel.
        After the workers complete, it synchronizes with other devices using a
        barrier.
        """

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.timepoint_done.wait()
            
            self.nr_iter = len(self.device.scripts) / 8
            
            if self.nr_iter == 0:
                scriptthreads = []
                for (script, location) in self.device.scripts:
                    scriptthreads.append(MiniDeviceThread(self.device, script, location, neighbours))
                
                for _, thread in enumerate(scriptthreads):
                    thread.start()
                
                for _, thread in enumerate(scriptthreads):
                    thread.join()
            
            
            else:
                count = 0
                size = 8
                for _ in range(self.nr_iter):
                    scriptthreads = []
                    for idx in range(count, size):
                        script = self.device.scripts[idx][0]
                        location = self.device.scripts[idx][1]
                        scriptthreads.append(MiniDeviceThread(self.device, script, location, neighbours))
                    
                    for _, thread in enumerate(scriptthreads):
                        thread.start()
	                
                    for _, thread in enumerate(scriptthreads):
                        thread.join()
                    count = count + 8
                    if size + 8 > len(self.device.scripts):
                        size = len(self.device.scripts) - size
                    else:
                        size = size + 8
            
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
