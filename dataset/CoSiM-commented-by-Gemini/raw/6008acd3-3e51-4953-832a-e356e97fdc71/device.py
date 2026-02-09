"""
@file device.py
@brief Defines a device model for a distributed simulation using a global lock.

This file implements a `Device` class where all script executions across the
entire system are serialized by a single, global lock. The root device (ID 0)
is responsible for creating and distributing this lock and a shared barrier to
all other devices.
"""

from threading import Event, Thread, Lock
import barrier

class Device(object):
    """
    Represents a device in the simulation, using a global lock for all data operations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): A dictionary of the device's sensor readings.
            supervisor: The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.my_lock = None
        self.timepoint_done = Event()
        self.bariera = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared global barrier and a global lock.

        This method, when called on the root device (ID 0), creates a single lock
        and a single barrier that are then shared by all devices in the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            lent = len(devices)
            bariera = barrier.ReusableBarrier(lent)
            my_lock = Lock()
            
            # Invariant: Distribute the same barrier and lock instances to all devices.
            for device in devices:
                device.bariera = bariera
            for device in devices:
                device.my_lock = my_lock

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script: The script object to run.
            location: The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data. Note: this method itself is not synchronized.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data. Note: this method itself is not synchronized.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop.

        It processes scripts serially, using a global lock to ensure that only
        one script in the entire system can run at any given time.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to signal script assignment is complete.
            self.device.timepoint_done.wait()
            
            # Block Logic: Process all scripts serially.
            for (script, location) in self.device.scripts:
                # Pre-condition: Acquire the single global lock. This serializes
                # all script executions across all devices.
                self.device.my_lock.acquire()
                script_data = []
                
                # Gather data from neighbors and self.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Invariant: Data is gathered and ready for script execution.
                if script_data != []:
                    
                    result = script.run(script_data)
		    
                    # Propagate results back to all relevant devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                
                # Release the global lock.
                self.device.my_lock.release()
	    
            self.device.timepoint_done.clear()
            # Wait at the barrier for all devices to finish their timepoint.
            self.device.bariera.wait()