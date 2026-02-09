"""
A device simulation framework with a deadlock-prone distributed locking scheme.

This module implements a device simulation that uses a shared, class-level
barrier for synchronization. The core script-processing logic attempts to
implement a distributed lock by sequentially acquiring a lock from every
neighboring device. This approach is fundamentally flawed and will lead to
deadlocks. The simulation also lacks a second barrier to synchronize the end of
a time step, causing further concurrency issues.
"""

from threading import *
from barrier import *


class Device(object):
    """
    Represents a single device, each with its own lock, participating in a
    globally synchronized simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

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
        self.lock = Lock()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the shared barrier with the total number of devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        DeviceThread.barr.set_th(len(devices))


    def assign_script(self, script, location):
        """
        Assigns a script and signals completion of the assignment phase.

        Args:
            script: The script object to execute.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
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
    The main control thread for a device, containing a flawed distributed
    locking protocol.
    
    This class holds a single, shared barrier instance for all threads.
    """
    # A single barrier instance shared across all DeviceThread instances.
    barr = ReusableBarrierCond()

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        

    def run(self):
        """
        The main execution loop, organized into unsynchronized timepoints.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Synchronize all threads at the beginning of the timepoint.
            DeviceThread.barr.wait()
            
            # Wait for the supervisor to signal that scripts are assigned.
            self.device.timepoint_done.wait()
            
            # Block Logic: Process each script with a flawed locking scheme.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # CRITICAL: The thread attempts to acquire a lock from each neighbor
                # sequentially. This will lead to a distributed deadlock if two
                # threads attempt to acquire locks from each other.
                for device in neighbours:
                    device.lock.acquire()
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                if script_data != []:
                    result = script.run(script_data)
                    
                    # The locks are released one by one, but not in a 'finally'
                    # block, making the operation unsafe.
                    for device in neighbours:
                        device.lock.release()
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
            
            # The lack of a second barrier here means fast threads can loop around
            # and cause a deadlock at the barrier wait at the top of the loop.
            self.device.timepoint_done.clear()