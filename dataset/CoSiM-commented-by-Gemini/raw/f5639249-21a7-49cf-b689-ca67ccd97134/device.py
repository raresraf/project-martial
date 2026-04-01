"""
This module simulates a network of devices for distributed sensor data processing.

It defines classes for devices, their execution threads, and synchronization primitives
to model a system where devices execute scripts based on their own and their
neighbors' sensor data in discrete time steps.
"""
from threading import *


class Device(object):
    """Represents a single device in the simulated distributed network.

    Each device has a unique ID, local sensor data, and a supervisor to determine
    its neighbors. It executes assigned scripts and synchronizes with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor values.
            supervisor (Supervisor): An object that provides neighborhood information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.lock_data = Lock()
        self.lock_location = []
        self.time_barrier = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources for all devices in the simulation.

        This method is intended to be called on a single device (e.g., device 0)
        to initialize shared locks and barriers for the entire system.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            # A reusable barrier for synchronizing all devices at the end of a time step.
            self.time_barrier = ReusableBarrierSem(len(devices)) 

            for device in devices:
                device.time_barrier = self.time_barrier

            loc_num = 0

            # Determines the maximum location ID to size the location locks array.
            for device in devices:
                for location in device.sensor_data:
                    loc_num = max(loc_num, location) 
            for i in range(loc_num + 1):
                self.lock_location.append(Lock()) 

            for device in devices:
                device.lock_location = self.lock_location 

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device.

        Args:
            script (Script): The script object to be executed. If None, it signals
                         the end of a timepoint.
            location (int): The location context for the script execution.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location.

        Args:
            location (int): The location for which to retrieve data.

        Returns:
            The sensor data at the given location, or None if not available.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a given location, with thread safety.

        Args:
            location (int): The location to update.
            data: The new data value.
        """
        with self.lock_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main execution thread for a Device.

    This thread orchestrates the device's lifecycle, waiting for simulation
    timepoints, executing scripts, and synchronizing with other devices.
    """

    def __init__(self, device):
        """Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main loop of the device thread."""
        while True:
            slaves = []
            
            # Retrieves the device's neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Waits for the signal that a timepoint has concluded and scripts are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() 

            
            # Spawns a SlaveThread for each assigned script.
            for (script, location) in self.device.scripts:
                slave = SlaveThread(script, location, neighbours, self.device) 
                slaves.append(slave)
                slave.start()

            # Waits for all slave threads to complete their execution.
            for i in range(len(slaves)):
                slaves.pop().join()

            # Synchronizes with all other devices at the end of the time step.
            self.device.time_barrier.wait() 

class SlaveThread(Thread):
    """A thread to execute a single script on a device."""
    def __init__(self, script, location, neighbours, device):
        """Initializes the SlaveThread.

        Args:
            script (Script): The script to execute.
            location (int): The data location the script operates on.
            neighbours (list): A list of neighboring Device objects.
            device (Device): The parent device.
        """

        Thread.__init__(self, name="Slave Thread of Device %d" % device.device_id)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """Executes the script."""
        device = self.device
        script = self.script
        location = self.location
        neighbours = self.neighbours
        
        data = device.get_data(location)
        input_data = []
        this_lock = device.lock_location[location]

        if data is not None:
            input_data.append(data) 

        # Acquires a lock for the specific location to ensure data consistency.
        with this_lock: 
            # Gathers data from all neighbors at the same location.
            for neighbour in neighbours:
                temp = neighbour.get_data(location) 

                if temp is not None:
                    input_data.append(temp)

            if input_data != []: 
                # Executes the script with the collected data.
                result = script.run(input_data) 

                # Propagates the result to all neighbors and the local device.
                for neighbour in neighbours:
                    neighbour.set_data(location, result) 

                device.set_data(location, result) 


class ReusableBarrierSem():
    """A reusable barrier implemented using semaphores.

    This barrier synchronizes a fixed number of threads at a point, and can be
    used multiple times. It uses a two-phase signaling mechanism.
    """
    
    def __init__(self, num_threads):
        """Initializes the reusable barrier.

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
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread to arrive releases all waiting threads for phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire()
         
    def phase2(self):
        """Second phase of the barrier synchronization, allowing reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread to arrive releases all waiting threads for phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            self.count_threads1 = self.num_threads
         
        self.threads_sem2.acquire()
