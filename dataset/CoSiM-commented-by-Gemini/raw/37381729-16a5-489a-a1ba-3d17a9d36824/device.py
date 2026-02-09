"""
A device simulation framework featuring a complex and flawed distributed locking scheme.

This module implements a device simulation where each worker thread (`MyThread`)
attempts to acquire a lock from every one of its neighboring devices before
processing data. This distributed, unordered lock acquisition strategy is
inherently prone to deadlocks. Furthermore, the lock release mechanism is not
guaranteed, making the system's stability highly precarious.
"""

from threading import Event, Thread, Lock
from reusableBarrier import ReusableBarrier

class MyThread(Thread):
    """
    A worker thread that executes a single script using a distributed locking protocol.
    
    This thread attempts to synchronize with its neighbors by acquiring a
    location-specific lock from each one before reading data. This approach is
    highly susceptible to deadlocks.
    """
    def __init__(self, d, location, script, neighbours):
        Thread.__init__(self)
        self.d = d
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        The main execution logic, containing the flawed locking protocol.
        """
        script_data = []

        # Block Logic: The thread attempts to acquire a lock for the target
        # location from every single neighbor sequentially. This is a classic
        # deadlock scenario if two threads try to acquire locks from each other
        # in a different order.
        for device in self.neighbours:
            keys = device.dictionar.keys()
            if self.location in keys:
                lock = device.dictionar[self.location]
                if lock is not None:
                    lock.acquire()

            # Data is read after each lock acquisition.
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.d.device.get_data(self.location)
        if data is not None:
            script_data.append(data)
        
        # Invariant: The script only runs if data was successfully gathered.
        if script_data != []:
            result = self.script.run(script_data)
            
            # Block Logic: Broadcasts the result and releases the previously
            # acquired locks. The release is not in a 'finally' block, so
            # if an error occurs before this point, the locks are never released.
            for device in self.neighbours:
                device.set_data(self.location, result)
                keys = device.dictionar.keys()
                if self.location in keys:
                    lock = device.dictionar[self.location]
                    if lock is not None:
                        lock.release()
            
            # Finally, acquire the parent device's own lock to set its data.
            self.d.device.lock.acquire()
            self.d.device.set_data(self.location, result)
            self.d.device.lock.release()

class Device(object):
    """
    Represents a single device, which creates its own set of locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance. Each device creates its own dictionary
        of location-specific locks.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's internal sensor data.
            supervisor: The central supervisor managing the device network.
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
        self.lock = Lock()
        self.dictionar = {}
        for location in self.sensor_data:
            if location != None:
                self.dictionar[location] = Lock()
            else:
                self.dictionar[location] = None

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier to all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script for execution in the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating timepoints.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop, organized into discrete timepoints."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Block Logic: Spawns one worker thread per assigned script.
            my_thread_list = []
            for (script, location) in self.device.scripts:
                my_thread = MyThread(self, location, script, neighbours)
                my_thread_list.append(my_thread)
                my_thread.start()
            
            # Wait for all worker threads to complete.
            for thread in my_thread_list:
                thread.join()

            # Invariant: All devices must synchronize at the barrier before the
            # next timepoint can begin.
            self.device.barrier.wait()