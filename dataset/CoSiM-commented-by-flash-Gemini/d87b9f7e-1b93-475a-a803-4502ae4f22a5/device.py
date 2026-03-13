"""
This module provides a simulation framework for a network of distributed devices.
It features a `Device` class representing each node, a `DeviceThread` for the main
control loop of each device, and a `RunScripts` class as the worker thread for
executing tasks. Synchronization across all devices is managed by a `ReusableBarrier`,
and location-specific data access is controlled by a complex lock-sharing mechanism.
"""


from threading import Thread, Event
from threading import Lock, Semaphore

class ReusableBarrier():
    """
    A reusable double-barrier synchronization primitive using semaphores.

    This implementation uses a two-phase (double turnstile) approach to allow a
    fixed number of threads to wait for each other at a synchronization point,
    and to be safely reused multiple times.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Public method for a thread to wait at the barrier."""
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """First phase (entry turnstile) of the barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads      
        self.threads_sem1.acquire()
    
    def phase2(self):
        """Second phase (exit turnstile) of the barrier, enables reusability."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads   
        self.threads_sem2.acquire()

class RunScripts(Thread):
    """
    A worker thread responsible for executing a single script on a device.
    
    It handles data aggregation from neighbors, script execution, and propagation
    of results, ensuring thread safety for a specific location via locks.
    """
    
    def __init__(self, device, location, script, neighbours):
        """
        Initializes the worker thread.

        Args:
            device (Device): The parent device this worker belongs to.
            location (str): The location context for the script.
            script (object): The script object to execute (must have a `run` method).
            neighbours (list): A list of neighboring `Device` objects.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    
    def run(self):
        """
        Core logic of the worker thread.

        Workflow:
        1. Acquire the location-specific lock to ensure exclusive access.
        2. Aggregate data from its own device and neighbors for that location.
        3. Execute the script with the aggregated data.
        4. Write the result back to its own device and all neighbors.
        5. Release the location lock.
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

class Device(object):
    """
    Represents a single device (or node) in the distributed simulation.
    
    Each device manages its own state, executes assigned scripts, and coordinates
    with other devices using shared synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of the device's internal data.
            supervisor (object): The central supervisor for fetching neighbors.
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
        # This list holds location-specific locks; its management is complex.
        self.location_lock = [None] * 200

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Establishes the shared barrier for synchronization among all devices.

        This method should be called on one device to create and distribute a
        `ReusableBarrier` to all other devices in the simulation.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        
        nr_devices = len(devices)
        
        if self.barrier is None:
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
        Assigns a script to the device and handles the complex lock-sharing logic.

        If a lock for the given `location` does not exist on this device, it will
        search other devices for an existing lock to "borrow". If none is found,
        it creates a new one. This ensures all devices use a single lock instance
        per location.

        Args:
            script (object): The script to be executed.
            location (str): The location context for the script.
        """
        lock_location = False

        if script is None:
            self.timepoint_done.set()



        else:
            
            self.scripts.append((script, location))
            if self.location_lock[location] is None:

                for device in self.devices:
                    if device.location_lock[location] is not None:
                        
                        self.location_lock[location] = device.location_lock[location]
                        lock_location = True
                        break

                if lock_location is False:
                    self.location_lock[location] = Lock()

            self.script_received.set()
            

    def get_data(self, location):
        """Retrieves data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main control thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control loop for a single `Device`.

    This thread orchestrates the device's lifecycle in the simulation, including
    running scripts and synchronizing with other devices at the end of each time step.
    """

    def __init__(self, device):
        """
        Initializes the control thread.

        Args:
            device (Device): The parent `Device` object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop of the device.

        Workflow per time step:
        1. Wait until the `timepoint_done` event is set, signaling all scripts
           for the current step have been assigned.
        2. Create a `RunScripts` worker thread for each assigned script.
        3. Start all worker threads.
        4. Wait for all worker threads to complete their execution.
        5. Clean up, clear the event, and wait at the global barrier for all
           other devices to finish the time step.
        """
        
        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            


            for (script, location) in self.device.scripts:
                thread = RunScripts(self.device, location, script, neighbours) 
                self.device.list_thread.append(thread)

            
            for thread_elem in self.device.list_thread:
                thread_elem.start()

            for thread_elem in self.device.list_thread:
                thread_elem.join()

            self.device.list_thread = []
            self.device.timepoint_done.clear()
            self.device.barrier.wait()