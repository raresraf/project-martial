"""
A device simulation framework using a condition-variable-based distributed lock.

This script implements a distributed device simulation where script execution on
specific locations is mutually exclusive across all devices. This is achieved
using a shared `threading.Condition` variable and a list that tracks active
locations, effectively creating a distributed locking mechanism. Synchronization
between time steps is handled by a reusable barrier.
"""

from threading import Event, Thread, Lock, Condition
import barrier

class Device(object):
    """
    Represents a single device in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: The initial sensor data for the device.
            supervisor: The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        
        # Synchronization primitives.
        self.bariera = barrier.ReusableBarrierCond(1)
        self.data_lock = Lock()
        self.script_lock = Lock()
        
        # Primitives for distributed location locking.
        self.locationcondition = Condition()
        self.locationlist = []

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization primitives for all devices.

        Args:
            devices: A list of all devices in the simulation.
        """
        
        if self.device_id is 0:
            # Device 0 creates and distributes the shared objects.
            self.bariera = barrier.ReusableBarrierCond(len(devices))
            for device in devices:
                device.bariera = self.bariera
                device.locationcondition = self.locationcondition
                device.locationlist = self.locationlist

    def assign_script(self, script, location):
        """
        Assigns a script to the device in a thread-safe manner.

        Args:
            script: The script to be executed.
            location: The location for the script's execution.
        """
        self.script_lock.acquire()

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()
        self.script_lock.release()

    def get_data(self, location):
        """
        Thread-safely retrieves data for a given location.

        Args:
            location: The location to retrieve data from.

        Returns:
            The data at the location, or None.
        """
        self.data_lock.acquire()
        value = self.sensor_data[location] if location in self.sensor_data\
                                           else None
        self.data_lock.release()
        return value

    def set_data(self, location, data):
        """
        Thread-safely sets data for a given location.

        Args:
            location: The location to set data at.
            data: The new data value.
        """
        self.data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.data_lock.release()

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The parent Device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.timepoint_done.wait()
            
            
            self.device.script_lock.acquire()

            # Create and run a thread for each script.
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
            
            self.device.bariera.wait()

class ScriptThread(Thread):
    """
    A thread that executes a single script, using a distributed lock for its location.
    """

    def __init__(self, device, script, location, neighbours, locationlist,
                 locationcondition):
        """
        Initializes the ScriptThread.

        Args:
            device: The parent Device object.
            script: The script to execute.
            location: The location for execution.
            neighbours: A list of neighboring devices.
            locationlist: The shared list of locked locations.
            locationcondition: The shared condition variable for location locking.
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
        Executes the script after acquiring the distributed lock for the location.
        """
        sem = 1
        
        /**
         * @brief Acquires a distributed lock for the script's location.
         * This loop attempts to "claim" the location by adding it to a shared list.
         * If the location is already claimed, the thread waits on a condition
         * variable until another thread releases a lock. This ensures that only
         * one thread can operate on a given location at a time across all devices.
         */
        while sem is 1:
            self.locationcondition.acquire()
            if self.location in self.locationlist:
                self.locationcondition.wait()
            else:
                self.locationlist.append(self.location)
                sem = 0
            self.locationcondition.release()

        # Once the lock is acquired, proceed with script execution.
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
        
        /**
         * @brief Releases the distributed lock.
         * The location is removed from the shared list, and all waiting threads
         * are notified, allowing them to re-check for their desired location.
         */
        self.locationcondition.acquire()
        self.locationlist.remove(self.location)
        self.locationcondition.notify_all()
        self.locationcondition.release()