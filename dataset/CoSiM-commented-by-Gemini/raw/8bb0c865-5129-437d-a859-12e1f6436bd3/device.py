# -*- coding: utf-8 -*-
"""
A multi-threaded simulation of a distributed device network that uses
intra-device parallelism to process script assignments.

This script models a system of interconnected devices that operate in synchronized
time steps. A master device (device 0) is responsible for creating and
distributing shared synchronization primitives (a reusable barrier and a set of
location-based locks) to all other devices. Within each time step, each device
can spawn multiple helper threads to process its assigned scripts in parallel,
with work being distributed among the main device thread and its helpers.
"""

from threading import Event, Thread, Lock
# The ReusableBarrierCond is assumed to be in a file named 'barrier.py'
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device (node) in the distributed system.

    Each device manages its own data and script assignments, and orchestrates its
    workload across a main control thread and several helper threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (Supervisor): An object that provides network topology info.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Events for managing the thread's state machine.
        self.timepoint_done = Event()
        self.barrier_set = Event()

        # script_dict groups assigned scripts by their target location.
        self.script_dict = {}
        # A dictionary of shared locks, one for each data location.
        self.location_lock_dict = {}
        self.barrier = None

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def set_synchronization(self, barrier, location_lock_dict):
        """
        Receives and sets the shared synchronization primitives from the master device.
        """
        self.barrier = barrier
        self.location_lock_dict = location_lock_dict
        self.barrier_set.set()

    def setup_devices(self, devices):
        """
        Initializes the entire network's synchronization fabric.

        This method is intended to be called on one device (e.g., device 0), which
        then acts as a master to create and distribute a shared barrier and a
        common set of location-based locks to all devices in the system.
        """
        # Block-Logic: Master device (id 0) creates and distributes synchronization objects.
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            location_lock_dict = {}
            
            # Create one lock for each unique location across all devices.
            for device in devices:
                for location in device.sensor_data.keys():
                    # The use of .has_key is a Python 2 idiom.
                    if location_lock_dict.has_key(location) == False:
                        location_lock_dict[location] = Lock()

            # Distribute the shared objects to all devices.
            for device in devices:
                device.set_synchronization(barrier, location_lock_dict)

    def assign_script(self, script, location):
        """
        Assigns a script to this device. A 'None' script signals the end of a time step.
        """
        if script is not None:
            # Group scripts by location for parallel processing.
            # The use of .has_key is a Python 2 idiom.
            if self.script_dict.has_key(location) == False:
                self.script_dict[location] = []
            self.script_dict[location].append(script)
        else:
            # A 'None' script wakes up the main device thread to start computation.
            self.timepoint_done.set()

    def get_data(self, location):
        """Safely retrieves data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Safely updates data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to finish."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a single Device, orchestrating parallel execution.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop, managing time steps and parallel work distribution.
        """
        # Wait until the master device has distributed the synchronization objects.
        self.device.barrier_set.wait()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the signal that all scripts for this time step are assigned.
            self.device.timepoint_done.wait()

            nr_locations = len(self.device.script_dict)
            # Limit parallelism to 8 helper threads.
            nr_threads = min(nr_locations, 8) 

            if nr_locations != 0:
                # Block-Logic: Distribute work among this thread and helper threads.
                threads = []
                # Create helper threads.
                for i in xrange(nr_threads - 1): # xrange is a Python 2 idiom.
                    threads.append(DeviceThreadHelper(self.device, i + 1,
                    	nr_locations, nr_threads, neighbours))
                for thread in threads:
                    thread.start()

                # Get this thread's share of the work using list slicing with a step.
                locations_list = self.device.script_dict.items()
                my_list = locations_list[0: nr_locations : nr_threads]

                # Process this thread's share of the work.
                for (location, script_list) in my_list:
                    for script in script_list:
                        script_data = []
                        # Acquire lock for this specific location.
                        self.device.location_lock_dict[location].acquire()
                        for device in neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)

                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data:
                            result = script.run(script_data)
                            # Write result back to all relevant devices.
                            for device in neighbours:
                                device.set_data(location, result)
                            self.device.set_data(location, result)
                        
                        self.device.location_lock_dict[location].release()

                # Wait for all helper threads to complete their work.
                for thread in threads:
                    thread.join()

            # Synchronize with all other devices before starting the next time step.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()


class DeviceThreadHelper(Thread):
    """
    A worker thread that processes a subset of a device's scripts.
    """

    def __init__(self, device, helper_id, num_locations, pace, neighbours):
        Thread.__init__(self)
        self.device = device
        self.my_id = helper_id
        self.num_locations = num_locations
        self.pace = pace
        self.neighbours = neighbours

    def run(self):
        """
        Processes an assigned slice of the script locations.
        """
        # Get this helper's share of the work using list slicing with a step.
        locations_list = self.device.script_dict.items()
        my_list = locations_list[self.my_id: self.num_locations : self.pace]

        # The execution logic is identical to the one in the main DeviceThread.
        for (location, script_list) in my_list:
            for script in script_list:
                script_data = []
                self.device.location_lock_dict[location].acquire()
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    result = script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                self.device.location_lock_dict[location].release()
