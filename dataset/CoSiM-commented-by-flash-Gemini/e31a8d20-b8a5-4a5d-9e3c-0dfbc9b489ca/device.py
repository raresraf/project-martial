"""
@file device.py
@brief Implements a simulated distributed device system with fine-grained data locking and semaphore-based synchronization.

This module defines the core components for simulating a network of devices.
It features concurrent script execution with explicit locking for each data location
to ensure data integrity. All devices synchronize their time-steps using a
`ReusableBarrierSem` to ensure consistent state progression across the simulation.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    @class Device
    @brief Represents an individual device in the simulated distributed system.

    Each device manages its own sensor data, receives and queues scripts for execution,
    and coordinates with a central supervisor. It utilizes a list of per-location
    `Lock` objects for granular data access control and a shared `ReusableBarrierSem`
    for time-step synchronization with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary holding the device's sensor data (e.g., location -> value).
        @param supervisor The supervisor object, responsible for providing neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Invariant: An Event to signal that new scripts have been assigned to the device.
        self.script_received = Event()

        # Invariant: A list to store assigned scripts, each paired with its target location.
        self.scripts = []

        # Invariant: A list that will hold `Lock` objects, each protecting a specific data location.
        # This list is populated and distributed during `setup_devices`.
        self.lock_locations = []

        # Invariant: An instance of the `ReusableBarrierSem` for time-step synchronization.
        # It's initialized with 0 and later re-initialized with the correct number of devices.
        self.barrier = ReusableBarrierSem(0)

        # Invariant: The dedicated thread responsible for this device's operational logic.
        self.thread = DeviceThread(self)

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs initial setup for the device and distributes shared resources across all devices.

        Block Logic:
        1. Only `device_id == 0` determines the total number of unique data locations across all devices.
        2. A global `ReusableBarrierSem` is initialized with the total number of devices.
        3. A master list of `Lock` objects (`self.lock_locations`) is created, one for each unique location.
        4. All devices are then assigned the global barrier and a reference to the shared `lock_locations` list.
        5. Finally, each device's operational thread is started.
        @param devices A list of all Device objects in the simulation.
        """
        # Block Logic: Device with ID 0 is responsible for initializing shared resources.
        # Invariant: The `barrier` is initialized here for all devices to share.
        barrier = ReusableBarrierSem(len(devices))

        if self.device_id == 0:
            nr_locations = 0
            # Block Logic: Determines the maximum location ID to correctly size `lock_locations`.
            for i in range(len(devices)):
                for location in devices[i].sensor_data.keys():
                    if location > nr_locations:
                        nr_locations = location
            nr_locations += 1 # Inline: Adjusts to be 1-indexed count for locations.

            # Block Logic: Creates a `Lock` object for each unique data location.
            for i in range(nr_locations):
                lock_location = Lock()
                self.lock_locations.append(lock_location)

        # Block Logic: Distributes the shared `barrier` and `lock_locations` to all devices.
        for i in range(len(devices)):
            devices[i].barrier = barrier # Inline: Assigns the shared barrier.
            # Block Logic: Assigns the shared list of per-location locks.
            for j in range(nr_locations):
                devices[i].lock_locations.append(self.lock_locations[j])
            devices[i].thread.start() # Inline: Starts the operational thread for each device.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specified location.
        If `script` is `None`, it signals that all scripts for the current timepoint have been assigned.
        @param script The script object to be executed, or `None`.
        @param location The data location associated with the script.
        """
        # Block Logic: Appends the script to the internal list or sets the `script_received` event.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: The appropriate per-location lock must be acquired by the calling `Worker` thread.
        @param location The key for the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        # Pre-condition: Checks if the requested location exists in `sensor_data`.
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.
        Note: The appropriate per-location lock must be held by the calling `Worker` thread.
        @param location The key for the sensor data.
        @param data The new data to set.
        """
        # Pre-condition: Checks if the location exists in `sensor_data` before updating.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its dedicated operational thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief The dedicated operational thread for a Device object, orchestrating time-step execution.

    This thread manages the device's main operational loop, including:
    1. Periodically checking for updated neighbor information from the supervisor.
    2. Waiting for `script_received` to be signaled, indicating scripts are ready for execution.
    3. Dispatching `Worker` threads for each assigned script concurrently.
    4. Waiting for all `Worker` threads to complete their tasks for the current timepoint.
    5. Synchronizing with all other DeviceThreads via the `ReusableBarrierSem`.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device object this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Invariant: The loop continues as long as the supervisor provides neighbor information.
        Each iteration represents a complete time-step in the simulation, from script readiness
        to synchronized completion.
        """
        workers = [] # Invariant: List to hold `Worker` thread instances for the current time-step.

        while True:
            # Block Logic: Retrieves current neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If `neighbours` is None, it signifies a shutdown or termination condition.
            if neighbours is None:
                break

            # Block Logic: Waits until the `script_received` event is set, indicating scripts are assigned and ready.
            self.device.script_received.wait()
            # Inline: Clears the event to allow it to be set again for the next time-step.
            self.device.script_received.clear()

            # Block Logic: Creates `Worker` threads for each assigned script.
            for (script, location) in self.device.scripts:
                workers.append(Worker(self.device, script,
                                        location, neighbours))

            # Block Logic: Starts all `Worker` threads concurrently.
            for i in range(len(workers)):
                workers[i].start()

            # Block Logic: Waits for all `Worker` threads to complete their tasks for this timepoint.
            for i in range(len(workers)):
                workers[i].join()

            workers = [] # Inline: Clears the list of workers for the next time-step.

            # Block Logic: All devices synchronize at the `ReusableBarrierSem`, ensuring all devices
            # have completed their local processing for the current time-step before proceeding.
            self.device.barrier.wait()


class Worker(Thread):
    """
    @class Worker
    @brief A worker thread responsible for executing a single script for a specific data location.

    This thread encapsulates the logic for acquiring the necessary locks, collecting
    sensor data from its device and neighbors, executing the assigned script, and
    then updating the data. It ensures data consistency through per-location locking.
    """

    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes a Worker thread.
        @param device The Device object this worker belongs to.
        @param script The script object to be executed.
        @param location The specific data location this worker is processing.
        @param neighbours A list of neighboring Device objects.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Inline: Renames the thread for debugging.
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def solve_script(self, script, location, neighbours):
        """
        @brief Contains the core logic for executing a script, including data acquisition and update.

        Block Logic:
        1. Acquires the appropriate per-location lock to ensure exclusive access to the data at `location`.
        2. Gathers `script_data` from its own device and neighboring devices.
        3. If `script_data` is available, executes the script.
        4. Updates the data on its own device and neighboring devices with the script's result.
        5. Releases the per-location lock.
        @param script The script object to be executed.
        @param location The data location to process.
        @param neighbours A list of neighboring Device objects.
        """
        # Block Logic: Acquires the specific lock for `location` to protect data access and modification.
        self.device.lock_locations[location].acquire()

        script_data = [] # Invariant: List to store collected data for the script.

        # Block Logic: Collects data from neighboring devices for the specified location.
        for device in neighbours:
            data = device.get_data(location)
            # Invariant: Only appends data if it's not None.
            if data is not None:
                script_data.append(data)

        # Block Logic: Collects data from its own device for the specified location.
        data = self.device.get_data(location)
        # Invariant: Only appends data if it's not None.
        if data is not None:
            script_data.append(data)

        # Pre-condition: If `script_data` is not empty, there is data to process.
        if script_data != []:
            # Block Logic: Executes the script with the collected data.
            result = script.run(script_data)

            # Block Logic: Updates the data on neighboring devices with the script's result.
            for device in neighbours:
                device.set_data(location, result)

            # Block Logic: Updates the data on its own device with the script's result.
            self.device.set_data(location, result)

        # Inline: Releases the specific lock for `location`.
        self.device.lock_locations[location].release()

    def run(self):
        """
        @brief The main execution method for the Worker thread. It simply calls `solve_script`.
        """
        self.solve_script(self.script, self.location, self.neighbours)
