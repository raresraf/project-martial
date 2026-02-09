"""
@file device.py
@brief Defines a device model with a flawed cross-device locking strategy.

@warning This implementation is critically flawed and highly prone to deadlocks.
         The `MyThread` worker attempts to acquire locks on a device and all of its
         neighbors simultaneously, which will lead to circular-wait deadlocks
         in almost any non-trivial scenario.
"""

from threading import Event, Thread, Lock
from reusableBarrier import ReusableBarrier

class MyThread(Thread):
    """
    A worker thread that executes a single script.
    
    @warning Its `run` method implements a locking strategy that is almost
             guaranteed to cause deadlocks between neighboring devices.
    """
    def __init__(self, d, location, script, neighbours):
        Thread.__init__(self)
        self.d = d # This is the parent DeviceThread instance.
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        script_data = []

        # Block Logic: Attempt to acquire a location-specific lock from each neighbor.
        # This will deadlock if two neighbors try to run scripts involving each
        # other at the same time.
        for device in self.neighbours:
            keys = device.dictionar.keys()
            if self.location in keys:
                lock = device.dictionar[self.location]
                if lock is not None:
                    lock.acquire()

            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.d.device.get_data(self.location)
        if data is not None:
            script_data.append(data)
        
        # Invariant: If not deadlocked, data is gathered and ready for execution.
        if script_data != []:
            result = self.script.run(script_data)
            
            # Block Logic: Write back results and release the locks on neighbors.
            for device in self.neighbours:
                device.set_data(self.location, result)
                keys = device.dictionar.keys()
                if self.location in keys:
                    lock = device.dictionar[self.location]
                    if lock is not None:
                        lock.release()
            
            # The local device's write is protected by a different, unrelated lock.
            self.d.device.lock.acquire()
            self.d.device.set_data(self.location, result)
            self.d.device.lock.release()

class Device(object):
    """
    Represents a device in the simulation. Each device instance creates and
    manages its own set of locks, which are not shared with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
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
        # Each device creates its own dictionary of locks, not shared.
        self.dictionar = {}
        for location in self.sensor_data:
            if location != None:
                self.dictionar[location] = Lock()
            else:
                self.dictionar[location] = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes a shared barrier to all devices."""
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to be executed."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. This read is not synchronized."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This write is not synchronized by this method."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

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
            
            # Wait for supervisor to signal that scripts are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            my_thread_list = []
            
            # Block Logic: Create and start a worker thread for each script.
            for (script, location) in self.device.scripts:
                my_thread = MyThread(self, location, script, neighbours)
                my_thread_list.append(my_thread)
                my_thread.start()
            
            # Wait for all (potentially deadlocked) threads to complete.
            for thread in my_thread_list:
                thread.join()

            # Synchronize with other devices before the next timepoint.
            self.device.barrier.wait()