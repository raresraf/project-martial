"""
A device simulation framework using a centralized lock management system.

This script models a network of distributed devices that execute scripts at
various locations. The simulation employs a centralized setup phase where one
device creates a lock for each unique location and a reusable barrier for
synchronization. These synchronization primitives are then distributed to all
other devices, ensuring a correct and robust concurrency model for script
execution.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single device in the simulated network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

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

        
        self.lock_locations = []

        
        self.barrier = ReusableBarrierSem(0)

        
        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs centralized setup of shared synchronization primitives.

        Executed by device 0, this method creates a lock for each unique sensor
        location and a reusable barrier for all devices. These are then
        distributed to all devices in the simulation.

        Args:
            devices: A list of all devices in the simulation.
        """
        
        barrier = ReusableBarrierSem(len(devices))

        
        if self.device_id == 0:
            nr_locations = 0

            # Determine the total number of unique locations.
            for i in range(len(devices)):
                for location in devices[i].sensor_data.keys():
                    if location > nr_locations:
                        nr_locations = location
            
            nr_locations += 1

            # Create a lock for each location.
            for i in range(nr_locations):
                lock_location = Lock()
                self.lock_locations.append(lock_location)

            # Distribute the barrier and locks to all devices.
            for i in range(len(devices)):
                
                devices[i].barrier = barrier

                for j in range(nr_locations):
                    devices[i].lock_locations.append(self.lock_locations[j])

                
                devices[i].thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        Args:
            script: The script to be executed.
            location: The location for the script's execution.
        """
        if script is not None:
            
            self.scripts.append((script, location))
        else:
            # Signal that all scripts for the timepoint have been received.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves data for a given location.

        Args:
            location: The location to retrieve data from.

        Returns:
            The data at the location, or None if not available.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets data for a given location.

        Args:
            location: The location to set data at.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

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

        It waits for scripts, creates a worker thread for each script, waits for
        them to complete, and then synchronizes at the global barrier.
        """
        workers = []

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for all scripts for the current timepoint to be assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Create a worker thread for each script.
            for (script, location) in self.device.scripts:
                workers.append(Worker(self.device, script,
                                        location, neighbours))

            
            for i in range(len(workers)):
                workers[i].start()

            
            for i in range(len(workers)):


                workers[i].join()

            
            workers = []

            
            # Synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()



class Worker(Thread):
    """
    A worker thread responsible for executing a single script.
    """

    def __init__(self, device, script, location, neighbours):
        """
        Initializes a Worker thread.

        Args:
            device: The parent Device object.
            script: The script to be executed.
            location: The location for the script's execution.
            neighbours: A list of neighboring devices.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def solve_script(self, script, location, neighbours):
        """
        Executes the script in a thread-safe manner using a location-specific lock.
        """
        
        # Acquire the lock for the specific location to ensure mutual exclusion.
        self.device.lock_locations[location].acquire()

        script_data = []

        # Gather data from neighbors.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Gather data from the parent device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data)

            
            for device in neighbours:
                device.set_data(location, result)

            self.device.set_data(location, result)

        
        self.device.lock_locations[location].release()

    def run(self):
        """The main entry point for the worker thread."""
        self.solve_script(self.script, self.location, self.neighbours)