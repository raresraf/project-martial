"""
@44ce9ba2-3993-4c29-b5c3-94b81a0ed5b0/device.py
@brief Implements a simulated device for a distributed sensor network, with concurrent script execution and location-based locking.
This module defines a `Device` that processes sensor data, interacts with neighbors,
and executes scripts. It features a `DeviceThread` that dispatches individual scripts
to `OneThread` instances for concurrent processing. Synchronization is handled by
a `ReusableBarrierSem` and a list of `Lock` objects, `block_location`, which protect
access to sensor data on a per-location basis.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, a shared barrier, and location-specific locks.
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
        self.thread = DeviceThread(self)
        self.barrier = None # Shared barrier for global time step synchronization.
        self.thread.start()
        self.block_location = None # List of locks, one for each unique data location.

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources like the barrier and location-specific locks among devices.
        Only the device with `device_id == 0` initializes these shared resources,
        which are then distributed to all other devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup, typically by device 0.
        """
        # Block Logic: Initializes shared synchronization primitives if this is the first device.
        if self.device_id == 0:
            # Initializes a shared reusable barrier with the total number of devices.
            self.barrier = ReusableBarrierSem(len(devices))
            locations = [] # Temporarily stores all unique locations across all devices.
            
            # Block Logic: Gathers all unique sensor data locations from all devices.
            for device in devices:
                for location in device.sensor_data:
                    if location is not None:
                        if location not in locations:
                            locations.append(location)
            # Block Logic: Creates a list of locks, one for each unique data location.
            self.block_location = []
            for _ in range(len(locations)): # Using range instead of xrange for Python 3 compatibility.
                self.block_location.append(Lock())
            # Block Logic: Distributes the shared barrier and location locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.block_location = self.block_location

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        If a script is provided, it's added to the queue and the `script_received` event is set.
        If no script (i.e., `None`) is provided, it signals that the timepoint is done.
        @param script: The script object to assign, or `None` to signal completion.
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
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
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

class OneThread(Thread):
    """
    @brief A dedicated thread for executing a single script for a specific data location.
    This thread is responsible for gathering data, running the script, and then
    propagating the results to relevant devices, ensuring thread-safe access to data
    through location-specific locks.
    """
    
    def __init__(self, myid, device, location, neighbours, script):
        """
        @brief Initializes an `OneThread` instance.
        @param myid: A unique identifier for this script execution thread.
        @param device: The parent `Device` instance for which the script is being run.
        @param location: The data location that the script operates on.
        @param neighbours: A list of neighboring `Device` instances.
        @param script: The script object to execute.
        """
        Thread.__init__(self)
        self.myid = myid # Unique ID for this thread.
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def run(self):
        """
        @brief The main execution logic for `OneThread`.
        Block Logic:
        1. Acquires the lock specific to the `location` to ensure exclusive access to that data.
        2. Collects data from neighboring devices and its own device for the specified `location`.
        3. Executes the assigned `script` if any data was collected.
        4. Propagates the script's `result` to neighboring devices and its own device.
        5. Releases the location-specific lock.
        Invariant: All operations on a given data `location` are atomic due to the `block_location` lock.
        """
        # Block Logic: Acquires a lock specific to the data location to prevent race conditions during script execution.
        with self.device.block_location[self.location]:
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
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                # Block Logic: Updates its own device's data with the script's result.
                self.device.set_data(self.location, result)

class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts concurrently via `OneThread` instances, and synchronizing with
    other device threads using a reusable barrier.
    Time Complexity: O(T * S_total * (N + D)) where T is the number of timepoints, S_total is the total number
    of scripts executed by the device, N is the number of neighbors, and D is the data retrieval/setting operations.
    The concurrency of `OneThread` helps mitigate the script execution time.
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
        3. Creates and starts an `OneThread` for each assigned script, allowing concurrent execution.
           Invariant: All scripts for the current timepoint are executed in parallel.
        4. Waits for all `OneThread` instances to complete.
        5. Clears the `timepoint_done` event for the next timepoint.
        6. Synchronizes with all other device threads using a shared barrier.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break
            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()

            
            threads = [] # List to keep track of currently running script execution threads.
            myid = 0 # Counter for assigning unique IDs to `OneThread` instances.
            # Block Logic: Iterates through assigned scripts, creating and starting a new `OneThread` for each.
            # Invariant: Each script is executed concurrently in its own `OneThread`.
            for (script, location) in self.device.scripts:
                thread = OneThread(myid, self.device, location, neighbours, script)
                threads.append(thread)
                thread.start()
                myid += 1
            # Block Logic: Waits for all `OneThread` instances to complete their execution.
            for thread in threads:
                thread.join()
            
            # Block Logic: Clears the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.barrier.wait()
