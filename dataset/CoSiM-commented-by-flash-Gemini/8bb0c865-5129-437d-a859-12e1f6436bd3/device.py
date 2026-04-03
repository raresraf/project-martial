"""
@8bb0c865-5129-437d-a859-12e1f6436bd3/device.py
@brief This script implements device behavior for a distributed system simulation,
focusing on data processing, synchronization, and parallel script execution across locations.
It coordinates device operations using a reusable barrier and location-specific locks.
Domain: Concurrency, Distributed Systems, Simulation, Thread Synchronization, Parallel Processing.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond # Assuming ReusableBarrierCond is defined in barrier.py


class Device(object):
    """
    @brief Represents a single device in the simulated distributed system.
    Each device manages its own sensor data, executes scripts, and coordinates
    with a supervisor and neighboring devices. This version utilizes a conditional
    reusable barrier and location-specific locks for synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data for various locations.
        @param supervisor: A reference to the supervisor object for coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Inline: Event to signal that the current timepoint's processing is complete.
        self.timepoint_done = Event()
        # Inline: Event to signal that the barrier and location locks have been set up.
        self.barrier_set = Event()
        # Inline: Dictionary to store scripts assigned to specific locations.
        self.script_dict = {}
        # Inline: Dictionary to store locks for different locations, preventing concurrent writes.
        self.location_lock_dict = {}
        # Inline: Reference to the reusable barrier for synchronizing with other devices.
        self.barrier = None

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        @return: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def set_synchronization(self, barrier, location_lock_dict):
        """
        @brief Sets the shared synchronization objects (barrier and location locks) for the device.
        @param barrier: The shared ReusableBarrierCond instance.
        @param location_lock_dict: A dictionary of shared Lock objects for each location.
        """
        self.barrier = barrier
        self.location_lock_dict = location_lock_dict
        self.barrier_set.set()


    def setup_devices(self, devices):
        """
        @brief Configures the device with a list of other devices in the system.
        Device 0 is responsible for initializing the shared barrier and location locks.
        @param devices: A list of all Device objects in the simulation.
        """
        # Block Logic: Only device 0 initializes the shared barrier and location locks.
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            location_lock_dict = {}
            
            # Block Logic: Initialize a unique Lock object for each unique sensor data location across all devices.
            for device in devices:
                for location in device.sensor_data.keys():
                    # Inline: Check if a lock for the current location already exists.
                    if location_lock_dict.has_key(location) == False:
                        location_lock_dict[location] = Lock()

            # Block Logic: Distribute the initialized barrier and location locks to all devices.
            for device in devices:
                device.set_synchronization(barrier, location_lock_dict)


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location for this device.
        Scripts for the same location are queued. If no script is provided, it signals
        the end of scripts for the current timepoint.
        @param script: The script object to be executed.
        @param location: The data location the script operates on.
        """
        if script is not None:
            # Pre-condition: Initialize list for location if it doesn't exist.
            if self.script_dict.has_key(location) == False:
                self.script_dict[location] = []
            # Inline: Add the script to the list for the specified location.
            self.script_dict[location].append(script)
        else:
            # Inline: Signal that the current timepoint's script assignment is done.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The location identifier for which to retrieve data.
        @return: The sensor data for the location, or None if the location is not present.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
        @param location: The location identifier for which to set data.
        @param data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data


    def shutdown(self):
        """
        @brief Shuts down the device by joining its main thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main thread responsible for a Device's operational loop.
    It continuously waits for scripts, executes them (potentially in parallel
    with helper threads), and synchronizes with other devices using a barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device: The Device object that this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main loop for the device thread. It continuously fetches neighbors,
        waits for timepoint signals, dispatches script execution to helper threads,
        and synchronizes with other devices via a barrier.
        """
        # Block Logic: Wait until the shared barrier and location locks have been set up.
        self.device.barrier_set.wait()

        # Block Logic: Continuous operational loop for the device.
        while True:
            # Block Logic: Fetch the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If supervisor returns None, it indicates simulation termination.
            if neighbours is None:
                break

            # Block Logic: Wait until the current timepoint's scripts are ready to be processed.
            self.device.timepoint_done.wait()

            # @var nr_locations: Number of unique locations for which scripts are assigned.
            # @var nr_threads: Number of helper threads to spawn for parallel script execution (max 8).
            nr_locations = len(self.device.script_dict)
            nr_threads = min(nr_locations, 8) 

            # Block Logic: Spawn helper threads to process scripts in parallel.
            if nr_locations != 0:
                # Inline: List to store helper threads.
                threads = []
                # Block Logic: Create 'nr_threads - 1' helper threads. The main thread will act as one processor.
                for i in xrange(nr_threads - 1): # xrange is used for efficiency in older Python versions
                    threads.append(DeviceThreadHelper(self.device, i + 1,
                    	nr_locations, nr_threads, neighbours))
                # Block Logic: Start all helper threads.
                for thread in threads:
                    thread.start()

                # Block Logic: The main device thread processes its assigned subset of locations.
                # Inline: Get a list of (location, script_list) tuples.
                locations_list = self.device.script_dict.items()
                # Inline: Determine the subset of locations for this main thread based on helper_id (0 implied) and pace.
                my_list = locations_list[0: nr_locations : nr_threads]


                # Block Logic: Process scripts for each assigned location in this thread's subset.
                for (location, script_list) in my_list:
                    # Block Logic: Execute each script for the current location.
                    for script in script_list:
                        script_data = []

                        # Block Logic: Acquire location lock to ensure exclusive access to data.
                        self.device.location_lock_dict[location].acquire()
                        # Block Logic: Collect data from neighboring devices.
                        for device in neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)

                        # Block Logic: Collect data from the current device.
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        # Pre-condition: Execute script only if there is data.
                        if script_data != []:
                            # Inline: Run the script and get the result.
                            result = script.run(script_data)

                            # Block Logic: Update data on neighboring devices.
                            for device in neighbours:
                                device.set_data(location, result)

                            # Block Logic: Update data on the current device.
                            self.device.set_data(location, result)

                        # Post-condition: Release the location lock.
                        self.device.location_lock_dict[location].release()


                # Block Logic: Wait for all helper threads to complete their execution.
                for thread in threads:
                    thread.join()

            # Block Logic: Synchronize with other devices and reset for the next timepoint.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()


class DeviceThreadHelper(Thread):
    """
    @brief A helper thread for a Device, responsible for processing a subset of locations
    assigned to the device. This allows parallel execution of scripts across different locations.
    """

    def __init__(self, device, helper_id, num_locations, pace, neighbours):
        """
        @brief Initializes a DeviceThreadHelper instance.
        @param device: The parent Device object.
        @param helper_id: A unique ID for this helper thread (used to determine its subset of locations).
        @param num_locations: Total number of locations with assigned scripts.
        @param pace: The step size for distributing locations among threads.
        @param neighbours: A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.device = device
        self.my_id = helper_id
        self.num_locations = num_locations
        self.pace = pace
        self.neighbours = neighbours

    def run(self):
        """
        @brief Executes scripts for its assigned subset of locations, collecting data from
        neighbors and itself, processing it, and updating relevant device data.
        """
        # Inline: Get the list of all (location, script_list) tuples.
        locations_list = self.device.script_dict.items()
        # Inline: Determine the subset of locations this helper thread is responsible for.
        my_list = locations_list[self.my_id: self.num_locations : self.pace]

        # Block Logic: Process scripts for each assigned location in this thread's subset.
        for (location, script_list) in my_list:
            # Block Logic: Execute each script for the current location.
            for script in script_list:
                script_data = []

                # Block Logic: Acquire location lock to ensure exclusive access to data.
                self.device.location_lock_dict[location].acquire()
                # Block Logic: Collect data from neighboring devices.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Block Logic: Collect data from the current device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: Execute script only if there is data.
                if script_data != []:
                    # Inline: Run the script and get the result.
                    result = script.run(script_data)

                    # Block Logic: Update data on neighboring devices.
                    for device in self.neighbours:
                        device.set_data(location, result)

                    # Block Logic: Update data on the current device.
                    self.device.set_data(location, result)

                # Post-condition: Release the location lock.
                self.device.location_lock_dict[location].release()
