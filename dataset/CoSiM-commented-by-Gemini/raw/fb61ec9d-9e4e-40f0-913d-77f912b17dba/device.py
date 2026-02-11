"""
This module implements a device simulation for a concurrent system.

It features a complex, nested locking strategy in its worker threads and a
critically flawed local locking mechanism in the main Device class. The threading
model is also inefficient, creating a new thread for each script in every
timepoint.
"""

from threading import Event, Thread, Lock, Semaphore
from ReusableBarrier import ReusableBarrier


class Device(object):
    """
    Represents a single device in the simulation.

    This class contains a flawed locking implementation with separate locks for
    get and set operations, which does not protect against race conditions.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []

        
        # Flawed locking: separate locks for getter and setter provide no
        # protection against read-write or write-write race conditions.
        self.lock_setter = Lock()
        self.lock_getter = Lock()
        self.lock_assign = Lock()

        
        self.barrier = None
        self.location_lock = {}

        
        # A semaphore to throttle the number of concurrently processing scripts
        # for this device to 8.
        self.semaphore = Semaphore(8)


        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.

        The leader device (ID 0) creates a shared barrier and a shared
        dictionary of locks, which are then distributed to all devices before
        their main threads are started.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))

            
            for device in devices[:]:
                for loc in device.sensor_data.keys():
                    if loc not in self.location_lock:
                        self.location_lock[loc] = Lock()

            
            for device in devices[:]:
                device.barrier = self.barrier
                device.location_lock = self.location_lock
                
                device.thread.start()


    def assign_script(self, script, location):
        """Assigns a script to the device."""

        with self.lock_assign:

            if script is not None:
                self.scripts.append((script, location))
            else:
                self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data, using a flawed, getter-only lock.
        
        CRITICAL FLAW: This lock does not prevent a concurrent write from
        another thread calling `set_data`, which uses a different lock.
        """

        with self.lock_getter:

            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        """
        Sets sensor data, using a flawed, setter-only lock.

        CRITICAL FLAW: This lock does not prevent a concurrent read from
        another thread calling `get_data`, which uses a different lock.
        """

        with self.lock_setter:

            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class ScriptThread(Thread):
    """
    A worker thread for executing a single script.
    
    It uses a nested locking scheme to ensure both location-based exclusivity
    and device-level concurrency throttling.
    """

    def __init__(self, device_thread, script, location, neighbours):
        
        Thread.__init__(self)
        self.script = script
        self.device_thread = device_thread


        self.location = location
        self.neighbours = neighbours

    def run(self):
        """The main execution logic for the worker."""
        # 1. Acquire the shared lock for this specific location to prevent
        # other scripts from operating on the same location.
        self.device_thread.device.location_lock[self.location].acquire()

        # 2. Acquire the device-local semaphore to count towards the device's
        # concurrency limit of 8 processing threads.
        self.device_thread.device.semaphore.acquire()

        script_data = []
        
        for device in self.neighbours:


            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device_thread.device.set_data(self.location, result)

        # Release locks in reverse order of acquisition.
        self.device_thread.device.semaphore.release()
        self.device_thread.device.location_lock[self.location].release()




class DeviceThread(Thread):
    """
    The main control thread for the device.
    
    This thread implements an inefficient model where it creates a new thread
    for every script in every timepoint.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop."""
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break



            # Wait for a signal that all scripts for the timepoint have been assigned.
            self.device.script_received.wait()
            script_threads = []

            
            # Inefficiently create a new thread for every assigned script.
            for (script, location) in self.device.scripts:
                
                thread = ScriptThread(self, script, location, neighbours)
                script_threads.append(thread)
                thread.start()

            # Wait for all the newly created threads to complete.
            for thread in script_threads:
                thread.join()

            self.device.script_received.clear()

            
            # Wait at the global barrier for all other devices to finish.
            self.device.barrier.wait()
