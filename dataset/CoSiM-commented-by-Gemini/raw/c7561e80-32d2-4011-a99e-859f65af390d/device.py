"""
This module implements a simulation of a networked device for a concurrent
system, likely for tasks involving distributed data processing or sensor networks.

The implementation is for Python 2 and relies on threading Events, Locks, and
a semaphore-based barrier for synchronization between devices.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device operates in its own thread, executes assigned scripts, and
    synchronizes with other devices using shared primitives established during a
    setup phase.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor (object): The central supervisor managing the device network.
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
        self.block_location = None
    def __str__(self):
        """Returns a human-readable string representation of the device."""
        return "Device %d" % self.device_id
    def setup_devices(self, devices):
        """
        Initializes shared synchronization objects for a list of devices.

        This method uses a leader-based approach, where the device with ID 0
        is responsible for creating and distributing a shared barrier and a
        set of location-based locks to all other devices in the network.
        """


        # The device with ID 0 acts as the leader for the setup process.
        if self.device_id == 0:
            
            # Create a semaphore-based barrier for all participating devices.
            self.barrier = ReusableBarrierSem(len(devices))
            locations = []
            
            # Aggregate all unique data locations from all devices.
            for device in devices:
                for location in device.sensor_data:
                    if location is not None:


                        if location not in locations:
                            locations.append(location)
            # Create a list of locks, one for each unique location.
            self.block_location = []
            for _ in xrange(len(locations)):
                self.block_location.append(Lock())
            # Distribute the shared barrier and location locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.block_location = self.block_location
    def assign_script(self, script, location):
        """
        Assigns a script to the device for later execution.

        Args:
            script (object): The script to be executed.
            location (int): The index of the location for data processing.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()
    def get_data(self, location):
        """Retrieves data for a given location if it exists."""
        return self.sensor_data[location] if location in self.sensor_data else None
    def set_data(self, location, data):
        """Updates data for a given location if it exists."""
        if location in self.sensor_data:
            self.sensor_data[location] = data
    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()
class OneThread(Thread):
    """A worker thread to execute a single script at a specific location."""
    def __init__(self, myid, device, location, neighbours, script):
        Thread.__init__(self)
        self.myid = myid
        self.device = device
        self.location = location


        self.neighbours = neighbours
        self.script = script
    def run(self):
        """
        Executes the script run logic.

        It acquires a lock for the specific location, gathers data from itself
        and its neighbors, runs the script, and distributes the result back.
        """
        with self.device.block_location[self.location]:
            script_data = []
            
            # Gather data from neighboring devices for the specified location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Include the device's own data.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                # Execute the script with the collected data.
                result = self.script.run(script_data)

                
                # Update the data on all neighbors and the current device.
                for device in self.neighbours:


                    device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)

class DeviceThread(Thread):
    """The main control thread for a device's lifecycle."""
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
    def run(self):
        """
        The main loop for the device, coordinating script execution per timepoint.
        """
        while True:
            
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break
            # Wait for a signal that the timepoint is done and scripts can be run.
            self.device.timepoint_done.wait()

            
            threads = []
            myid = 0
            # For each assigned script, create and start a worker thread.
            for (script, location) in self.device.scripts:
                thread = OneThread(myid, self.device, location, neighbours, script)
                threads.append(thread)
                thread.start()
                myid += 1
            # Wait for all worker threads to complete their execution.
            for thread in threads:
                thread.join()
            
            # Reset the timepoint event and wait at the global barrier for all devices.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
