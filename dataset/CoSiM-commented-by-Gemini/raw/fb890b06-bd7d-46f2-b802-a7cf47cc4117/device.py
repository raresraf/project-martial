"""
This module implements a device simulation for a concurrent system.

This implementation is characterized by several unusual and flawed design
patterns, including a buggy barrier setup, a convoluted peer-to-peer mechanism
for sharing locks, and an inefficient thread-per-script execution model.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    """
    Represents a single device in the simulation.
    
    This class contains flawed setup logic and an unconventional method for
    sharing locks between devices.
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
        self.devices = []
        self.locks = {}
        self.lock_used = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the barrier for all devices.

        BUG: This method is called by every device. Each time it's called, it
        creates a new barrier and overwrites the barrier on all other devices.
        As a result, only the barrier created by the *last* device to call
        this method will actually be used by the system.
        """
        
        barrier = ReusableBarrierSem(len(devices))

        
        for device in devices:
            self.devices.append(device)
            device.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script and uses a peer-to-peer search to share a lock.

        This method implements a convoluted mechanism to ensure a shared lock
        for a given location. It iterates through all other known devices to
        see if one of them has already created a lock for this location. If so,
        it copies the reference. If not, it creates a new one.
        """
        if script is not None:
            self.scripts.append((script, location))
            
            # Search other devices to see if a lock for this location already exists.
            for device in self.devices:
                if device.locks.get(location) is not None:
                    self.locks[location] = device.locks[location]
                    self.lock_used = 1
                    break

            
            # If no lock was found after checking all peers, create a new one.
            if self.lock_used is None:
                self.locks[location] = Lock()

            self.lock_used = None
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class ScriptThread(Thread):
    """A worker thread that executes a single script."""

    def __init__(self, device, script, location, neighbours):
        
        Thread.__init__(self)


        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """Executes the script, using the shared location lock."""
        with self.device.locks[self.location]:
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


class DeviceThread(Thread):
    """
    The main control thread for the device.
    
    This thread implements an inefficient model where it creates a new thread
    for every script in every timepoint.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script_threads = []

    def run(self):
        """The main execution loop."""
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            # Inefficiently create, start, and join a new thread for each script.
            for (script, location) in self.device.scripts:
                thread = ScriptThread(self.device, script, location, neighbours)
                self.script_threads.append(thread)

            for thread in self.script_threads:
                thread.start()
            for thread in self.script_threads:
                thread.join()
            
            # Clear the list of threads for the next timepoint.
            self.script_threads = []

            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
