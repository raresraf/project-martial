"""
This module implements a multi-threaded simulation of a network of devices.
It uses a custom synchronization mechanism based on a shared list and a
Condition variable to ensure that only one script per location is executed
at a time across the entire system.
"""

from threading import Event, Thread, Lock, Condition
import barrier

class Device(object):
    """
    Represents a device in the simulation. Each device runs in its own thread
    and executes scripts that process sensor data.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id: Unique ID for the device.
            sensor_data: A dictionary of the device's sensor data.
            supervisor: The supervisor object for the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        
        # A barrier for synchronizing all devices at the end of a timepoint.
        self.bariera = barrier.ReusableBarrierCond(1)
        
        self.data_lock = Lock()
        self.script_lock = Lock()
        
        # A condition variable and a shared list to control access to locations.
        self.locationcondition = Condition()
        self.locationlist = []

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization objects for all devices.
        This should be called by a single device at the start.
        """
        if self.device_id is 0:
            self.bariera = barrier.ReusableBarrierCond(len(devices))
            for device in devices:
                device.bariera = self.bariera
                device.locationcondition = self.locationcondition
                device.locationlist = self.locationlist

    def assign_script(self, script, location):
        """Assigns a script to the device for execution."""
        self.script_lock.acquire()
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()
        self.script_lock.release()

    def get_data(self, location):
        """
        Gets sensor data for a given location in a thread-safe manner.
        """
        self.data_lock.acquire()
        value = self.sensor_data[location] if location in self.sensor_data else None
        self.data_lock.release()
        return value

    def set_data(self, location, data):
        """
        Sets sensor data for a given location in a thread-safe manner.
        """
        self.data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.data_lock.release()

    def shutdown(self):
        """Shuts down the device thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main thread of execution for a device."""

    def __init__(self, device):
        """
        Initializes the device thread.
        Args:
            device: The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main loop for the device thread. It coordinates script execution and synchronization.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.timepoint_done.wait()
            
            self.device.script_lock.acquire()

            # Create and start a thread for each script.
            nodes = []
            for (script, location) in self.device.scripts:
                nodes.append(ScriptThread(self.device, script, location,\
                             neighbours, self.device.locationlist,\
                             self.device.locationcondition))
            for j in xrange(len(self.device.scripts)):
                nodes[j].start()
            for j in xrange(len(self.device.scripts)):
                nodes[j].join()
            
            self.device.timepoint_done.clear()
            self.device.script_lock.release()
            
            # Wait at the barrier for all devices to finish the timepoint.
            self.device.bariera.wait()

class ScriptThread(Thread):
    """A thread that executes a single script."""

    def __init__(self, device, script, location, neighbours, locationlist,\
                 locationcondition):
        """
        Initializes the script thread.
        Args:
            device, script, location, neighbours: standard parameters.
            locationlist: A shared list of currently processed locations.
            locationcondition: A Condition variable to synchronize access to the locationlist.
        """
        Thread.__init__(self, name="Service Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.locationlist = locationlist
        self.locationcondition = locationcondition

    def run(self):
        """
        Executes the script after acquiring a 'lock' on the location.
        """
        sem = 1
        
        # This loop implements a custom lock for the location. A thread tries to
        # add its location to a shared list. If the location is already there,
        # it waits. This ensures that only one thread can process a given
        # location at a time across the whole system.
        while sem is 1:
            self.locationcondition.acquire()
            if self.location in self.locationlist:
                self.locationcondition.wait()
            else:
                self.locationlist.append(self.location)
                sem = 0
            self.locationcondition.release()

        # Collect data from the device and its neighbors.
        script_data = []
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        if script_data != []:
            # Run the script and update the data.
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)
        
        # Release the 'lock' on the location.
        self.locationcondition.acquire()
        self.locationlist.remove(self.location)
        self.locationcondition.notify_all()
        self.locationcondition.release()
