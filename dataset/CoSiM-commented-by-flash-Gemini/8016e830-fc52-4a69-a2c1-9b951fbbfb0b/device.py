"""
@8016e830-fc52-4a69-a2c1-9b951fbbfb0b/device.py
@brief Implements a simulated device for a distributed sensor network, with concurrent script execution and dynamic, location-based locking.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` that manages task distribution to a pool of "minion" threads
(created dynamically) for parallel execution. Synchronization is handled by a shared
`ReusableBarrierCond` for global time-step coordination, and a dictionary of `Lock` objects
(`locks`) provides per-location data protection across devices.
"""

from threading import Event, Thread, Lock
from multiprocessing import cpu_count
from barrier import ReusableBarrier # Assumed to contain ReusableBarrier class.

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
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.thread = DeviceThread(self) # The dedicated thread for this device.
        
        self.locks = [] # List of Locks for location-specific data access.
        
        # Initialized with 0 threads, then reconfigured in `setup_devices`.
        self.barrier = ReusableBarrier(0)

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
        which are then distributed to all other devices. It also starts the `DeviceThread` for each device.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        if self.device_id == 0:

            # Block Logic: Determines the maximum location index across all devices to size the lock list.
            nr_locations = 0
            for i in range(len(devices)): # Changed xrange to range for Python 3 compatibility.
                nr_locations = max(nr_locations, max(devices[i].sensor_data.keys()))

            # Block Logic: Initializes a list of Locks, one for each potential location.
            for i in range(nr_locations + 1): # Changed xrange to range for Python 3 compatibility.
                self.locks.append(Lock())

            # Block Logic: Initializes the shared `ReusableBarrier` with the total number of devices.
            barrier = ReusableBarrier(len(devices))

            # Block Logic: Distributes the initialized shared barrier and `locks` list to all devices.
            # Then starts each device's main `DeviceThread`.
            for i in range(len(devices)): # Changed xrange to range for Python 3 compatibility.
                
                devices[i].barrier = barrier
                
                for j in range(nr_locations + 1): # Changed xrange to range for Python 3 compatibility.
                    devices[i].locks.append(self.locks[j])

            
            for i in range(len(devices)): # Changed xrange to range for Python 3 compatibility.
                devices[i].thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Signals that a script has been received, or that a timepoint is done if no script.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Block Logic: Signals completion of the timepoint if no script is assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `Worker` thread will acquire the appropriate `locks[location]` before calling this method.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `Worker` thread will acquire the appropriate `locks[location]` before calling this method.
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
    executing scripts concurrently via `Worker` instances, and coordinating with
    other device threads using a shared `ReusableBarrier`.
    Time Complexity: O(T * S_total * (N * D_access + D_script_run)) where T is the number of timepoints,
    S_total is the total number of scripts executed by the device, N is the number of neighbors,
    D_access is data access time, and D_script_run is script execution time.
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
        2. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.
        3. Creates and starts `Worker` instances for each assigned script, allowing concurrent execution.
           Invariant: All scripts for the current timepoint are executed in parallel.
        4. Waits for all `Worker` instances to complete.
        5. Synchronizes with all other device threads using a shared `ReusableBarrier`.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        6. Clears the `timepoint_done` event for the next cycle.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()

            # Block Logic: Creates a list of `Worker` instances, one for each script, for concurrent execution.
            worker_list = []
            for (script, location) in self.device.scripts:
                worker_list.append(Worker(self.device,
                	location, script, neighbours))

           	# Block Logic: Starts all created `Worker` instances.
            for i in range(len(worker_list)): # Changed xrange to range for Python 3 compatibility.
                worker_list[i].start()

            # Block Logic: Waits for all initiated `Worker` instances to complete their execution.
            for i in range(len(worker_list)): # Changed xrange to range for Python 3 compatibility.
                worker_list[i].join()

            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.barrier.wait()

            # Block Logic: Clears the `timepoint_done` event for the next timepoint cycle.
            self.device.timepoint_done.clear()


class Worker(Thread):
    """
    @brief A dedicated thread for executing a single script for a specific data location.
    This thread is responsible for gathering data, running the script, and then
    propagating the results to relevant devices, ensuring thread-safe access to data
    through location-specific `Lock` objects.
    """
    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes a `Worker` instance.
        @param device: The parent `Device` instance for which the script is being run.
        @param location: The data location that the script operates on.
        @param script: The script object to execute.
        @param neighbours: A list of neighboring `Device` instances.
        """
        Thread.__init__(self, name="Worker")
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for `Worker`.
        Block Logic:
        1. Acquires the location-specific lock (`locks[self.location]`) to ensure exclusive access to that data.
        2. Collects data from neighboring devices and its own device for the specified `location`.
        3. Executes the assigned `script` if any data was collected.
        4. Propagates the script's `result` to neighboring devices and its own device.
        5. Releases the location-specific lock.
        Invariant: All data access and modification for a given `location` are protected by a shared `Lock`.
        """
        # Block Logic: Acquires the location-specific lock to ensure exclusive access to data at this `location`.
        self.device.locks[self.location].acquire()

        script_data = []

        # Block Logic: Collects data from neighboring devices for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Collects data from its own device for the specified location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script if any data was collected and propagates the result.
        if script_data != []:
            
            result = self.script.run(script_data)

            # Block Logic: Updates neighboring devices with the script's result.
            for dev in device.neighbours:
                dev.set_data(self.location, result)

            # Block Logic: Updates its own device's data with the script's result.
            device.set_data(self.location, result)

        # Block Logic: Releases the location-specific lock after all data operations for this script are complete.
        device.locks[self.location].release()