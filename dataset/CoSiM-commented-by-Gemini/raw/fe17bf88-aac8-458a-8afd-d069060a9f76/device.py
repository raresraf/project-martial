"""
This module defines classes for a simulated distributed device network.

It includes a `Device` class that runs concurrently, processes data with scripts,
and synchronizes with other devices using a reusable barrier. This is likely part
of a simulation framework for sensor networks or distributed computing environments.
"""

from threading import Thread, Event
from threading import Lock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This barrier implementation uses two phases to allow threads to wait for each
    other to reach a certain point, and then be released, ready for another
    synchronization event.
    """
    
    
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have arrived.
        """
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """The first phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread arrived, release all waiting threads for phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads      
        self.threads_sem1.acquire()
    
    def phase2(self):
        """The second phase of the barrier synchronization, allows reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread arrived, release all waiting threads for phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads   
        self.threads_sem2.acquire()

class RunScripts(Thread):
    """
    A thread to execute a script on a device and its neighbors.

    This thread handles data gathering from neighboring devices at a specific
    location, running a script on the collected data, and then propagating
    the results back to the devices.
    """

    def __init__(self, device, location, script, neighbours):
        """
        Initializes the script-running thread.

        Args:
            device (Device): The main device object.
            location (int): The location index for data operations.
            script (object): The script object to be run, must have a `run` method.
            neighbours (list): A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    
    def run(self):
        """The main execution method for the thread."""
        
        # Acquire a lock for the given location to ensure data consistency.
        self.device.location_lock[self.location].acquire()

        script_data = []
        
        # Gather data from all neighbors at the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        # Gather data from the current device as well.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            # Execute the script with the collected data.
            result = self.script.run(script_data)
            
            

            # Distribute the result back to the neighbors and the current device.
            for device in self.neighbours:
                device.set_data(self.location, result)
                
            self.device.set_data(self.location, result)

        
        # Release the lock for the location.
        self.device.location_lock[self.location].release()

class Device(object):
    """
    Represents a single device in a distributed network simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data, keyed by location.
            supervisor (object): A supervisor object that manages the network.
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
        # Pre-allocate a list for location-based locks.
        self.location_lock = [None] * 200

    def __str__(self):
        """String representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's knowledge of other devices and initializes the barrier.
        """
        
        
        nr_devices = len(devices)
        
        if self.barrier is None:
            # Create a shared barrier for all devices if it doesn't exist.
            barrier = ReusableBarrier(nr_devices)
            self.barrier = barrier

            for device in devices:
                if device.barrier is None:


                    device.barrier = barrier

        
        for device in devices:
            if device is not None:
                self.devices.append(device)


    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location.
        """
        
        lock_location = False

        if script is None:
            # A None script signals the end of a timepoint.
            self.timepoint_done.set()



        else:
            
            self.scripts.append((script, location))
            if self.location_lock[location] is None:

                # If no lock exists for this location, try to find a shared one or create a new one.
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        
                        self.location_lock[location] = device.location_lock[location]
                        lock_location = True
                        break

                if lock_location is False:
                    self.location_lock[location] = Lock()

            self.script_received.set()
            

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        
        self.thread.join()

class DeviceThread(Thread):
    """
    The main thread of execution for a Device.
    """

    def __init__(self, device):
        """
        Initializes the device's main thread.

        Args:
            device (Device): The device this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop for the device, runs scripts and synchronizes.
        """
        
        
        while True:

            
            # Get the list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown.
                break

            # Wait until all scripts for the current timepoint are assigned.
            self.device.timepoint_done.wait()

            
            # Create and store script-running threads.
            for (script, location) in self.device.scripts:
                thread = RunScripts(self.device, location, script, neighbours) 
                self.device.list_thread.append(thread)

            
            # Start all script-running threads.
            for thread_elem in self.device.list_thread:
                thread_elem.start()

            # Wait for all script-running threads to complete.
            for thread_elem in self.device.list_thread:
                thread_elem.join()

            self.device.list_thread = []
            self.device.timepoint_done.clear()
            # Synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()