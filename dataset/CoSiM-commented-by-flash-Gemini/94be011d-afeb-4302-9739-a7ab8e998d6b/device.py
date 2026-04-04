

"""
@94be011d-afeb-4302-9739-a7ab8e998d6b/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices with parallel script execution per device.

This module defines the core components for simulating a network of sensor devices,
each capable of executing scripts, managing local sensor data, and interacting
with a central supervisor. This version utilizes a `DeviceThread` that spawns
`DeviceThreadHelper` threads to process different locations concurrently for a device.
Synchronization across devices is handled by a `ReusableBarrierCond`.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate with neighbors under the
guidance of a supervisor.

Classes:
- Device: Represents a single simulated sensor device.
- DeviceThread: Manages the lifecycle and operation of a Device, including spawning
                and coordinating `DeviceThreadHelper` instances.
- DeviceThreadHelper: A worker thread responsible for executing scripts for a subset of locations.

Domain: Distributed Systems Simulation, Concurrent Programming, Parallel Processing, Sensor Networks.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. This version
    uses a dedicated `DeviceThread` which, in turn, can spawn multiple
    `DeviceThreadHelper` instances to process scripts for different locations
    in parallel. Synchronization across devices is handled by a `ReusableBarrierCond`.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data, keyed by location.
        @param supervisor: A reference to the central supervisor managing all devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Event to signal when the device has completed processing for the current timepoint.
        self.timepoint_done = Event()
        # Event to signal when the global barrier and location locks have been set up.
        self.barrier_set = Event()
        # Dictionary to store scripts assigned to this device, keyed by location.
        self.script_dict = {}
        # Dictionary to hold locks for each location, ensuring exclusive access during data manipulation.
        self.location_lock_dict = {}
        # Reference to the global synchronization barrier.
        self.barrier = None

        # Spawns and starts the main thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def set_synchronization(self, barrier, location_lock_dict):
        """
        @brief Sets the global synchronization barrier and location-specific locks for the device.

        This method is called by the supervisor (or device 0) to distribute the
        shared synchronization primitives to all devices.

        @param barrier: The ReusableBarrierCond instance used for global device synchronization.
        @param location_lock_dict: A dictionary of Lock objects, one for each unique location.
        """
        self.barrier = barrier
        self.location_lock_dict = location_lock_dict
        self.barrier_set.set()


    def setup_devices(self, devices):
        """
        @brief Orchestrates the setup of synchronization mechanisms across all devices.

        This method is intended to be called by a single device (typically device 0)
        to initialize the global barrier and per-location locks, and then distribute
        them to all other devices.

        Pre-condition: This method should only be called by a designated device (e.g., device_id == 0).
        Invariant: After execution, all devices will have the same barrier and location_lock_dict.

        @param devices: A list of all Device instances in the simulation.
        """
        if self.device_id == 0:
            # Initialize a reusable barrier for all devices.
            barrier = ReusableBarrierCond(len(devices))
            location_lock_dict = {}
            
            # Block Logic: Iterates through all devices and their sensor data to aggregate all unique locations.
            # For each unique location, a new Lock object is created to manage concurrent access.
            for device in devices:
                for location in device.sensor_data.keys():
                    if location_lock_dict.has_key(location) == False:
                        location_lock_dict[location] = Lock()
            # Block Logic: Distributes the initialized barrier and location locks to all devices.
            for device in devices:
                device.set_synchronization(barrier, location_lock_dict)


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed for a specific location on this device.

        If `script` is None, it signals that the device has received all scripts for
        the current timepoint and is ready to process.

        @param script: The script object to be executed, or None to signal completion.
        @param location: The location pertinent to the script execution.
        """
        if script is not None:
            # Pre-condition: `script` is not None, indicating a script needs to be assigned.
            # Invariant: The script will be added to the list of scripts for the specified location.
            if self.script_dict.has_key(location) == False:
                self.script_dict[location] = []
            self.script_dict[location].append(script)
        else:
            # Pre-condition: `script` is None.
            # Invariant: The `timepoint_done` event is set, signaling readiness for processing.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The location for which to retrieve data.
        @return: The sensor data for the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The location for which to set data.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data


    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's thread.

        Waits for the device's main thread to complete its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the execution lifecycle for a single Device instance in a dedicated thread.

    This thread is responsible for fetching scripts, coordinating parallel execution
    of scripts for different locations using `DeviceThreadHelper` instances,
    and synchronizing with other devices at each timepoint.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        @param device: The Device instance that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Pre-condition: The synchronization barrier and locks must be set for the device.
        Invariant: The device continuously processes timepoints, executes scripts,
                   and synchronizes with other devices until a shutdown signal is received.
        """
        # Block Logic: Waits until the global barrier and location locks have been properly set up.
        self.device.barrier_set.wait()

        # Main simulation loop.
        while True:
            # Block Logic: Fetches the current neighbors of this device from the supervisor.
            # This allows for dynamic network topology changes between timepoints.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Pre-condition: `neighbours` is None, indicating a termination signal from the supervisor.
                # Invariant: The loop breaks, leading to thread termination.
                break

            # Block Logic: Waits for the supervisor to signal that scripts for the current
            # timepoint have been assigned to all devices.
            self.device.timepoint_done.wait()

            nr_locations = len(self.device.script_dict)
            # Limits the number of helper threads to a maximum of 8 or the number of locations, whichever is smaller.
            nr_threads = min(nr_locations, 8) 

            # Block Logic: If there are locations with scripts to process, it initializes and starts
            # DeviceThreadHelper instances for parallel processing.
            if nr_locations != 0:
                # Pre-condition: `nr_locations` > 0, meaning there are scripts to execute.
                # Invariant: Helper threads are spawned to process locations in parallel.
                threads = []
                # Spawns `nr_threads - 1` helper threads, as the main thread will also process a subset of locations.
                for i in xrange(nr_threads - 1):
                    threads.append(DeviceThreadHelper(self.device, i + 1,
                    	nr_locations, nr_threads, neighbours))
                for thread in threads:
                    thread.start()

                # Block Logic: The main device thread also processes its own subset of locations.
                # It iterates through assigned scripts for each location, acquires the location lock,
                # retrieves data from neighbors and itself, executes the script, and updates data.
                locations_list = self.device.script_dict.items()
                # Inline: Distributes locations among the main thread and helper threads.
                my_list = locations_list[0: nr_locations : nr_threads]


                for (location, script_list) in my_list:
                    # Pre-condition: `location` and `script_list` are from the device's assigned script dictionary.
                    # Invariant: The script is executed, and relevant data is updated while holding the location lock.
                    for script in script_list:
                        script_data = []

                        # Critical Section: Acquires a lock for the current location to prevent race conditions.
                        self.device.location_lock_dict[location].acquire()
                        # Block Logic: Gathers data from neighboring devices for the current location.
                        for device in neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)

                        # Gathers data from its own sensor_data for the current location.
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data != []:
                            # Executes the script with the collected data.
                            result = script.run(script_data)

                            # Block Logic: Updates data on neighboring devices with the script's result.
                            for device in neighbours:
                                device.set_data(location, result)

                            # Updates its own data with the script's result.
                            self.device.set_data(location, result)

                        # Releases the lock for the current location.
                        self.device.location_lock_dict[location].release()


                # Block Logic: Waits for all spawned helper threads to complete their execution.
                for thread in threads:
                    thread.join()

            # Block Logic: Synchronizes with all other DeviceThreads using a global barrier.
            # This ensures all devices have finished processing their locations before proceeding to the next timepoint.
            self.device.barrier.wait()
            # Resets the timepoint_done event, preparing for the next timepoint.
            self.device.timepoint_done.clear()



class DeviceThreadHelper(Thread):
    """
    @brief A worker thread responsible for executing a subset of scripts for specific locations
           on a device in parallel.

    Multiple instances of this class can be spawned by a `DeviceThread` to
    distribute the workload of script execution across different locations.
    """

    def __init__(self, device, helper_id, num_locations, pace, neighbours):
        """
        @brief Initializes the DeviceThreadHelper.

        @param device: The parent Device instance.
        @param helper_id: A unique identifier for this helper thread.
        @param num_locations: The total number of locations with scripts to process.
        @param pace: The stride value used to determine which locations this helper thread processes.
        @param neighbours: A list of neighboring Device instances to interact with.
        """
        Thread.__init__(self)
        self.device = device
        self.my_id = helper_id
        self.num_locations = num_locations
        self.pace = pace
        self.neighbours = neighbours

    def run(self):
    	"""
        @brief The main execution method for the DeviceThreadHelper.

        Pre-condition: The device's `script_dict` contains scripts, and `location_lock_dict`
                       and `neighbours` are properly initialized.
        Invariant: This thread processes its assigned subset of locations, executing scripts
                   and updating data while respecting location-specific locks.
        """
        # Block Logic: Selects a subset of locations for this helper thread to process
        # based on its `helper_id` and `pace`.
        locations_list = self.device.script_dict.items()
        # Inline: Uses slicing with a step (`pace`) to evenly distribute locations among helper threads.
        my_list = locations_list[self.my_id: self.num_locations : self.pace]

        # Block Logic: Iterates through the assigned locations and their scripts.
        # For each script, it acquires the location lock, gathers data, executes the script,
        # updates data on itself and neighbors, and then releases the lock.
        for (location, script_list) in my_list:
            # Pre-condition: `location` and `script_list` are from the helper thread's assigned locations.
            # Invariant: The script is executed, and relevant data is updated while holding the location lock.
            for script in script_list:
                script_data = []

                # Critical Section: Acquires a lock for the current location to prevent race conditions.
                self.device.location_lock_dict[location].acquire()
                # Block Logic: Gathers data from neighboring devices for the current location.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gathers data from its own sensor_data for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Executes the script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Updates data on neighboring devices with the script's result.
                    for device in self.neighbours:
                        device.set_data(location, result)

                    # Updates its own data with the script's result.
                    self.device.set_data(location, result)

                # Releases the lock for the current location.
                self.device.location_lock_dict[location].release()

