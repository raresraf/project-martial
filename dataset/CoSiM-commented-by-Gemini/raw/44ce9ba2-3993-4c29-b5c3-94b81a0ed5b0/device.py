"""
@file device.py
@brief Defines a device model for a distributed simulation with location-based locking.

This file implements a `Device` that executes scripts in a simulated network.
It uses a dedicated thread class, `OneThread`, for each script execution, and
employs a set of locks, one for each unique data location, to manage concurrency.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a device in the simulation, managing sensor data and script execution.

    It coordinates with a supervisor and other devices, using a shared barrier and
    a set of location-specific locks for synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): A dictionary mapping locations to sensor values.
            supervisor: The central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barrier = None
        self.thread.start()
        # This will hold the list of locks, one for each location.
        self.block_location = None
        
    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id
        
    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization primitives.

        Called by the root device (ID 0), this method creates a shared barrier
        and a list of locks corresponding to each unique location found across
        all devices. These are then distributed to all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Pre-condition: Only the root device (ID 0) should execute this block.
        if self.device_id == 0:
            
            self.barrier = ReusableBarrierSem(len(devices))
            locations = []
            
            # Block Logic: First, gather all unique locations from all devices.
            for device in devices:
                for location in device.sensor_data:
                    if location is not None:
                        if location not in locations:
                            locations.append(location)

            # Block Logic: Create one lock for each unique location.
            self.block_location = []
            for _ in xrange(len(locations)):
                self.block_location.append(Lock())

            # Invariant: Distribute the shared barrier and location locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.block_location = self.block_location

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed.

        Args:
            script: The script object to run.
            location: The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that all scripts for the timepoint have been assigned.
            self.timepoint_done.set()
            
    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None
        
    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data
            
    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()
        
class OneThread(Thread):
    """
    A worker thread responsible for executing a single script.

    Each instance of this class handles one script, ensuring that data access
    for the script's target location is serialized via a lock.
    """
    def __init__(self, myid, device, location, neighbours, script):
        """
        Initializes the script-execution thread.

        Args:
            myid (int): A unique ID for this thread instance.
            device (Device): The parent device.
            location: The location context for the script.
            neighbours (list): A list of neighboring devices.
            script: The script to execute.
        """
        Thread.__init__(self)
        self.myid = myid
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script
        
    def run(self):
        """
        Executes the script for the given location with proper locking.
        """
        # The 'with' statement ensures the location-specific lock is acquired
        # before data access and released afterwards.
        with self.device.block_location[self.location]:
            script_data = []
            
            # Block Logic: Gather data from neighbors and the local device.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Invariant: script_data contains all data for the location from
            # the local device and its neighborhood.
            if script_data != []:
                
                result = self.script.run(script_data)

                # Block Logic: Propagate the result back to all devices in the neighborhood.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)

class DeviceThread(Thread):
    """
    The main control thread for a Device.

    This thread orchestrates the execution of scripts for each timepoint by
    spawning `OneThread` workers.
    """
    def __init__(self, device):
        """
        Initializes the main device thread.

        Args:
            device (Device): The parent device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
    def run(self):
        """
        The main simulation loop for the device.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals simulation end.
                break

            # Waits for the supervisor to assign all scripts for the timepoint.
            self.device.timepoint_done.wait()

            
            threads = []
            myid = 0
            # Block Logic: For each assigned script, create and start a dedicated
            # worker thread (`OneThread`) to execute it.
            for (script, location) in self.device.scripts:
                thread = OneThread(myid, self.device, location, neighbours, script)
                threads.append(thread)
                thread.start()
                myid += 1
            
            # Block Logic: Wait for all worker threads for this timepoint to complete.
            for thread in threads:
                thread.join()
            
            # Clear the event for the next timepoint and synchronize at the barrier.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()