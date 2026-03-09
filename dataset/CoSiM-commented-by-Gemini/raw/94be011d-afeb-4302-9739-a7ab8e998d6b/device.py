"""
This module implements a distributed device simulation with a unique threading model.
In each time step, a main device thread dynamically creates a pool of temporary
helper threads to process work in parallel. This is generally less efficient than
using a persistent thread pool.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a device in the simulation. It manages sensor data and orchestrates
    script execution by spawning temporary worker threads in each time step.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The device's local sensor data.
            supervisor: The central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.timepoint_done = Event()  # Signals that scripts for a timepoint are ready.
        self.barrier_set = Event()  # Signals that sync objects have been set up.
        # Scripts are grouped by the location they operate on.
        self.script_dict = {}
        # A shared dictionary of locks, one for each data location.
        self.location_lock_dict = {}
        self.barrier = None # The main barrier for synchronizing all devices.

        # The single, persistent control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def set_synchronization(self, barrier, location_lock_dict):
        """
        Receives and sets the shared synchronization objects from the setup coordinator.
        """
        self.barrier = barrier
        self.location_lock_dict = location_lock_dict
        # Signal that the barrier and locks are now available.
        self.barrier_set.set()


    def setup_devices(self, devices):
        """
        Initializes synchronization objects for the entire simulation.

        Device 0 acts as the coordinator, creating a shared barrier and a dictionary
        of locks for all unique data locations, then distributing them to all devices.
        """
        # Block Logic: The device with ID 0 performs the global setup.
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            location_lock_dict = {}
            
            # Aggregate all unique locations from all devices.
            for device in devices:
                for location in device.sensor_data.keys():
                    if location_lock_dict.has_key(location) == False:
                        # Create one lock for each unique location.
                        location_lock_dict[location] = Lock()
            
            # Distribute the shared objects to all devices.
            for device in devices:
                device.set_synchronization(barrier, location_lock_dict)


    def assign_script(self, script, location):
        """
        Assigns a script to a location for the current timepoint.

        Args:
            script: The script to execute. If None, it signals the end of script
                    assignment for the current timepoint.
            location: The location the script targets.
        """
        if script is not None:
            # Group scripts by their target location.
            if self.script_dict.has_key(location) == False:
                self.script_dict[location] = []
            self.script_dict[location].append(script)
        else:
            # A None script signals that the device can start processing this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data


    def shutdown(self):
        """Shuts down the device by joining its main control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. In each time step, it spawns,
    manages, and joins a temporary pool of DeviceThreadHelper workers.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        # Wait until the master device (device 0) has set up the barrier and locks.
        self.device.barrier_set.wait()

        while True:
            # Get neighbors for the current timepoint from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the signal that all scripts for this timepoint have been assigned.
            self.device.timepoint_done.wait()

            nr_locations = len(self.device.script_dict)
            # Determine the size of the temporary thread pool for this timepoint.
            nr_threads = min(nr_locations, 8) 

            if nr_locations != 0:
                # --- Dynamic Worker Thread Creation ---
                # A new pool of helper threads is created for each timepoint.
                threads = []
                for i in xrange(nr_threads - 1):
                    threads.append(DeviceThreadHelper(self.device, i + 1,
                    	nr_locations, nr_threads, neighbours))
                for thread in threads:
                    thread.start()

                # --- Static Work Distribution ---
                # The work (locations) is partitioned between this main thread and the helpers.
                locations_list = self.device.script_dict.items()
                # This main thread takes the first slice of the work.
                my_list = locations_list[0: nr_locations : nr_threads]

                # This thread processes its own assigned work.
                for (location, script_list) in my_list:
                    for script in script_list:
                        script_data = []
                        
                        # Acquire lock for the specific location.
                        self.device.location_lock_dict[location].acquire()
                        # --- Critical Section ---
                        for device in neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data != []:
                            result = script.run(script_data)
                            # Update data on all relevant devices.
                            for device in neighbours:
                                device.set_data(location, result)
                            self.device.set_data(location, result)
                        # --- End Critical Section ---
                        self.device.location_lock_dict[location].release()

                # Wait for all temporary helper threads to complete.
                for thread in threads:
                    thread.join()

            # All devices synchronize at the main barrier before the next timepoint.
            self.device.barrier.wait()
            # Reset the event for the next cycle.
            self.device.timepoint_done.clear()



class DeviceThreadHelper(Thread):
    """
    A temporary worker thread created to process a fraction of the work for a
    single timepoint. It is created and destroyed within each simulation step.
    """
    def __init__(self, device, helper_id, num_locations, pace, neighbours):
        Thread.__init__(self)
        self.device = device
        self.my_id = helper_id
        self.num_locations = num_locations
        self.pace = pace
        self.neighbours = neighbours

    def run(self):
        # Determine this thread's static slice of the work.
        locations_list = self.device.script_dict.items()
        my_list = locations_list[self.my_id: self.num_locations : self.pace]

        # The execution logic is identical to the work-processing part of DeviceThread.
        for (location, script_list) in my_list:
            for script in script_list:
                script_data = []
                self.device.location_lock_dict[location].acquire()
                # --- Critical Section ---
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    result = script.run(script_data)

                    for device in self.neighbours:
                        device.set_data(location, result)

                    self.device.set_data(location, result)
                # --- End Critical Section ---
                self.device.location_lock_dict[location].release()