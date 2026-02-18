"""
@7f1a2cf6-dba4-497a-a90c-5c755898e068/device.py
@brief Implements a simulated device for a distributed sensor network, with concurrent script execution using a dynamic thread pool.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` that manages task distribution to a pool of "minion" threads
(created dynamically) for parallel execution. Synchronization is handled by a shared
`ReusableBarrierCond` for global time-step coordination, and a dictionary of `Lock` objects
(`locks`) provides per-location data protection across devices.
"""

from threading import Event, Thread, Lock
from multiprocessing import cpu_count
from barrier import * # Assumed to contain ReusableBarrierCond class.


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, a shared barrier, and a dictionary of location-specific locks.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing the device's local sensor readings.
        @param supervisor: The supervisor object responsible for managing the overall simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal that a script has been assigned.
        self.scripts = [] # List to store assigned scripts.
        self.timepoint_done = None # Shared `ReusableBarrierCond` for global time-step synchronization.
        self.thread = DeviceThread(self) # The main thread for this device.
        self.neighbours = [] # List to store neighboring devices.
        self.locks = None # Dictionary to hold `Lock` objects for each sensor data location.
        # Determines the maximum number of concurrent "minion" threads, at least 8 or the CPU count.
        self.max_minions = max(8, cpu_count())

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (global barrier and location-specific locks) among all devices.
        Only the device with `device_id == 0` is responsible for initializing these resources,
        which are then distributed to all other devices. Also starts the `DeviceThread` for each device.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        if self.device_id == 0:
            
            # Block Logic: Initializes the shared global `ReusableBarrierCond` with the total number of devices.
            barrier = ReusableBarrierCond(len(devices))

            # Block Logic: Creates a dictionary of `Lock` objects for each unique sensor data location across all devices.
            locks = {}
            for dev in devices:
                for pair in dev.sensor_data:
                    if not pair in locks:
                        locks[pair] = Lock()

            # Block Logic: Distributes the initialized shared barrier and `locks` dictionary to all devices.
            # Then starts each device's main `DeviceThread`.
            for dev in devices:
                dev.timepoint_done = barrier # The barrier acts as the "timepoint_done" signal for all.
                dev.locks = locks
                dev.thread.start() # Start the main device thread.


    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        If no script is provided, it signals `script_received`, indicating the completion of
        script assignment for the current timepoint.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: Signals that script assignments for the current timepoint are complete.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        thread (via `run_task`) will acquire the appropriate `locks[location]` before calling this method.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        thread (via `run_task`) will acquire the appropriate `locks[location]` before calling this method.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its graceful completion.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The dedicated main thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    distributing script execution among a pool of "minion" threads, and coordinating
    with other device threads using a shared `ReusableBarrierCond`.
    Time Complexity: O(T * S_total * (N * D_access + D_script_run) / P) where T is the number of timepoints,
    S_total is the total number of scripts executed by the device, N is the number of neighbors,
    D_access is data access time, and P is the size of the "minion" thread pool (`max_minions`).
    """
    

    def __init__(self, device):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Waits for the `script_received` event to be set, indicating that scripts are ready to be processed.
        3. Clears the `script_received` event.
        4. Distributes assigned scripts among a pool of `max_minions` (up to `cpu_count` or 8) dynamic threads.
        5. Starts all "minion" threads.
        6. Waits for all "minion" threads to complete their execution.
        7. Clears the list of "minion" threads.
        8. Synchronizes with all other device threads using a shared `ReusableBarrierCond` (acting as `timepoint_done`).
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        minions = [] # List to keep track of dynamically created "minion" threads.
        while True:
            # Block Logic: Fetches neighbor devices from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                break

            # Block Logic: Waits until the device has received all scripts for the current timepoint.
            self.device.script_received.wait()
            # Block Logic: Clears the `script_received` event for the next timepoint cycle.
            self.device.script_received.clear()

            # Block Logic: Initializes a dictionary to hold tasks, distributing them among `max_minions`.
            tasks = {}
            for i in range(self.device.max_minions): # Changed xrange to range for Python 3 compatibility.
                tasks[i] = []

            # Block Logic: Distributes scripts from `self.device.scripts` to the `tasks` dictionary
            # in a round-robin fashion for load balancing among "minion" threads.
            for i in range(len(self.device.scripts)): # Changed xrange to range for Python 3 compatibility.
                tasks[i % self.device.max_minions].append(self.device.scripts[i])

            # Block Logic: Creates and appends new `Thread` instances (minions) to `minions` list
            # for each group of tasks, targeting the `run_task` function.
            for i in range(self.device.max_minions): # Changed xrange to range for Python 3 compatibility.
                if len(tasks[i]) > 0:
                    minions.append(Thread(target=run_task, args=(self.device, tasks[i])))

            # Block Logic: Starts all dynamically created "minion" threads.
            for minion in minions:
                minion.start()

            # Block Logic: Waits for all initiated "minion" threads to complete their execution.
            for minion in minions:
                minion.join()

            # Block Logic: Clears the list of "minion" threads for the next timepoint.
            while len(minions) > 0:
                minions.remove(minions[0])

            # Block Logic: Synchronizes with other device threads using a shared barrier (`timepoint_done`),
            # ensuring all devices complete their processing before proceeding.
            self.device.timepoint_done.wait()

def run_task(device, tasks):
    """
    @brief Executes a list of tasks (scripts) for a specific device.
    This function is designed to be run by a "minion" thread. It iterates through
    its assigned tasks, acquires a location-specific lock for each, collects data,
    runs the script, and propagates results before releasing the lock.
    @param device: The `Device` instance to which these tasks belong.
    @param tasks: A list of `(script, location)` tuples to be executed.
    """
    for task in tasks:
        (script, location) = task
        script_data = []

        # Block Logic: Acquires the location-specific lock to ensure exclusive access to data at this `location`.
        device.locks[location].acquire()

        # Block Logic: Collects data from neighboring devices for the specified location.
        for dev in device.neighbours:
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Collects data from its own device for the specified location.
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script if any data was collected and propagates the result.
        if script_data != []:
            
            result = script.run(script_data)

            # Block Logic: Updates neighboring devices with the script's result.
            for dev in device.neighbours:
                dev.set_data(location, result)

            # Block Logic: Updates its own device's data with the script's result.
            device.set_data(location, result)

        # Block Logic: Releases the location-specific lock after all data operations for this script are complete.
        device.locks[location].release()