"""
This module implements a device simulation for a concurrent system.

Its architecture is defined by a main device thread that, in each timepoint,
creates a new pool of worker threads to execute scripts. Tasks are distributed
to these workers in a round-robin fashion. This model of creating and
destroying threads in a loop is generally inefficient.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single device in the simulation.
    
    It relies on a leader device (ID 0) to set up shared synchronization
    primitives like a barrier and a dictionary of location-based locks.
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
        self.locationlocks = {}
        # A device-local lock to protect its own sensor_data.
        self.lock = Lock()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.

        The leader device (ID 0) creates a shared barrier and a shared
        dictionary of locks for all unique data locations. These objects are
        then assigned to all other devices in the network.
        """
        bariera = ReusableBarrierSem(len(devices))
        locations = []
        
        
        for dev in devices:
            if (self.device_id == 0):
                dev.bariera = bariera
            # Aggregate all unique locations from all devices.
            for location in dev.sensor_data:
                if not location in locations:
                    locations.append(location)
        
        
        
        
        if (self.device_id == 0):
            # The leader creates and then distributes the shared lock dictionary.
            for location in locations:
                self.locationlocks[location] = Lock()
            for dev in devices:
                dev.locationlocks = self.locationlocks


    def assign_script(self, script, location):
        """Assigns a script to the device for execution."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data at a specific location in a thread-safe manner
        with respect to the device's own data structure.
        """
        
        
        
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class ScriptThread(Thread):
    """A worker thread responsible for executing a subset of scripts."""
    def __init__(self, device, scripts, locations, neighbours):
        
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts
        self.locations = locations
        self.neighbours = neighbours
    def run(self):
        """
        Executes the assigned scripts, acquiring the appropriate shared lock
        for each one.
        """
        i = 0
        for script in self.scripts:
            # Use the shared lock for the specific location.
            self.device.locationlocks[self.locations[i]].acquire()
            script_data = []
            for device in self.neighbours:
                data = device.get_data(self.locations[i])
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(self.locations[i])
            if data is not None:
                script_data.append(data)
            if script_data != []:
                result = script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.locations[i], result)
                self.device.set_data(self.locations[i], result)
            self.device.locationlocks[self.locations[i]].release()
            i += 1


class DeviceThread(Thread):
    """
    The main control thread for a device.
    
    This thread implements an inefficient model where it creates a new pool of
    worker threads for each timepoint and distributes work in a round-robin fashion.
    """

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop."""
        tlist = []
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            
            # --- Inefficient Threading Model ---
            # A new pool of 8 worker threads is created for every timepoint.
            for index in range(8):
                tlist.append(ScriptThread(self.device, [], [], neighbours))
            index = 0
            
            
            # Distribute scripts to the workers in a round-robin fashion.
            for (script, location) in self.device.scripts:
                tlist[index].scripts.append(script)
                tlist[index].locations.append(location)
                index = (index + 1) % 8
            
            # Start all the newly created worker threads.
            for thread in tlist:
                    thread.start()
            
            # Wait for all worker threads to complete.
            for thread in tlist:
                    thread.join()
            
            # Clear the list, destroying the thread objects.
            del tlist[:]
            
            
            # Wait at the global barrier for all other devices.
            self.device.bariera.wait()
