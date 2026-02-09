

"""
This module defines a distributed device simulation framework with a sophisticated
concurrency model for script execution.

It features a `Device` class representing a network node, a `ReusableBarrier` for
synchronizing threads across multiple phases, and a `DeviceThread` that spawns
individual `MyThread` instances for each script to be executed. This design allows
for parallel execution of multiple scripts within a single timepoint.
"""

from threading import Event, Thread, Lock, Semaphore
class ReusableBarrier(object):
    """
    A reusable barrier that synchronizes threads through two phases.

    This barrier implementation uses semaphores and a lock to ensure that all
    participating threads wait at the barrier before any of them are released.
    It is designed to be reusable for repeated synchronization points.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()           			
        self.threads_sem1 = Semaphore(0)            
        self.threads_sem2 = Semaphore(0)            
    def wait(self):
        """
        Causes the calling thread to wait through two synchronization phases.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    def phase(self, count_threads, threads_sem):
        """
        Executes a single synchronization phase.

        Threads calling this method will block until all threads have entered the phase.

        Args:
            count_threads (list): A list containing the count of remaining threads.
            threads_sem (Semaphore): The semaphore used for blocking and releasing threads.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for _ in range(self.num_threads):
                    threads_sem.release()
                    
                count_threads[0] = self.num_threads        
        threads_sem.acquire()        
                                    
class MyThread(Thread):
    """
    A dedicated thread for executing a single script on a device and its neighbors.
    """
    def __init__(self, neighbours, script, location, device):
        """
        Initializes the script execution thread.

        Args:
            neighbours (list): A list of neighboring devices.
            script (Script): The script to execute.
            location (int): The data location for the script.
            device (Device): The parent device.
        """
        Thread.__init__(self)
        self.neighbours = neighbours


        self.script = script
        self.location = location
        self.device = device
        self.script_data = []
    def run(self):
        """
        Gathers data from the device and its neighbors, executes the script,
        and propagates the result back to all involved devices.
        """
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                self.script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            self.script_data.append(data)
        if self.script_data != []:
            
            result = self.script.run(self.script_data)
            
            for device in self.neighbours:
                device.set_data(self.location, result)
                
            self.device.set_data(self.location, result)
        self.script_data = []
class Device(object):
    """
    Represents a device in a distributed network that can process sensor data.

    Each device runs in its own thread and can execute multiple scripts concurrently
    by spawning dedicated threads for each script. It synchronizes with other devices
    using a shared reusable barrier.
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
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for i in xrange(len(devices)):
                devices[i].barrier = barrier
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

class DeviceThread(Thread):
    """
    The main execution thread for a Device instance.

    This thread waits for a timepoint to begin, then spawns separate threads
    for each assigned script to execute them in parallel. After all script
    threads have completed, it synchronizes with other devices at a global barrier.
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
        1. Wait for the current timepoint to be triggered.
        2. For each assigned script, create and start a new `MyThread`.
        3. Wait for all `MyThread` instances to complete.
        4. Wait at a global barrier to ensure all devices have finished the timepoint.
        """
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            freds = []
            for (script, location) in self.device.scripts:
                fred = MyThread(neighbours, script, location, self.device)
                freds.append(fred)
                
            for i in freds:
                i.start()
            for i in freds:
                i.join()
            
            self.device.timepoint_done.clear()
            self.device.barrier.wait() 
