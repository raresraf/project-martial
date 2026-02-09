"""
@file device.py
@brief Defines a device model with multi-level and semaphore-based locking.

This file implements a simulation device that uses a variety of locking
mechanisms. The root device (ID 0) creates and distributes a shared barrier
and a map of location-specific locks. Each device also has three separate locks
for its getter, setter, and assignment methods. Script execution is handled
by `ScriptThread` workers, with concurrency limited by a semaphore.
"""

from threading import Event, Thread, Lock, Semaphore
from ReusableBarrier import ReusableBarrier


class Device(object):
    """
    Represents a device in the simulation, managing complex synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []

        # A dedicated lock for each core device operation.
        self.lock_setter = Lock()
        self.lock_getter = Lock()
        self.lock_assign = Lock()

        self.barrier = None
        self.location_lock = {}

        # A semaphore to limit the number of concurrently running script threads to 8.
        self.semaphore = Semaphore(8)

        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared barrier and location-specific locks.
        
        Executed by the root device (ID 0). It inspects all devices to discover
        all unique locations and creates a shared lock for each one.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))

            # Block Logic: Discover all unique locations and create a lock for each.
            for device in devices[:]:
                for loc in device.sensor_data.keys():
                    if loc not in self.location_lock:
                        self.location_lock[loc] = Lock()

            # Invariant: Distribute the shared barrier and lock map to all devices.
            for device in devices[:]:
                device.barrier = self.barrier
                device.location_lock = self.location_lock
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device, protected by a dedicated lock.
        """
        with self.lock_assign:
            if script is not None:
                self.scripts.append((script, location))
            else:
                # A None script signals the end of script assignment.
                self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data, protected by a dedicated lock."""
        with self.lock_getter:
            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        """Updates sensor data, protected by a dedicated lock."""
        with self.lock_setter:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class ScriptThread(Thread):
    """
    A worker thread that executes a single script, managing both location-based
    locking and semaphore-based concurrency limiting.
    """
    def __init__(self, device_thread, script, location, neighbours):
        Thread.__init__(self)
        self.script = script
        self.device_thread = device_thread
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """The main execution logic for the script."""
        # Pre-condition: Acquire the specific lock for the script's location.
        self.device_thread.device.location_lock[self.location].acquire()
        
        # Pre-condition: Wait for an available execution slot from the semaphore.
        self.device_thread.device.semaphore.acquire()

        script_data = []
        # Block Logic: Gather data from neighbors and self.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = self.script.run(script_data)
            # Propagate the result.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device_thread.device.set_data(self.location, result)

        # Release the semaphore slot and the location lock.
        self.device_thread.device.semaphore.release()
        self.device_thread.device.location_lock[self.location].release()


class DeviceThread(Thread):
    """
    The main control thread for a device.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            script_threads = []

            # Block Logic: Create and start a worker thread for each script.
            for (script, location) in self.device.scripts:
                thread = ScriptThread(self, script, location, neighbours)
                script_threads.append(thread)
                thread.start()
            
            # Wait for all script threads to complete.
            for thread in script_threads:
                thread.join()

            self.device.script_received.clear()

            # Synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()