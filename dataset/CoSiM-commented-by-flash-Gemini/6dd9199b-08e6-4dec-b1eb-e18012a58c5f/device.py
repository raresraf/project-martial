"""
@6dd9199b-08e6-4dec-b1eb-e18012a58c5f/device.py
@brief Implements a simulated device for a distributed sensor network, with sequential script execution and two-phase barrier synchronization.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` for operational logic and uses two `ReusableBarrier` instances
(`barrier1`, `barrier2`) for coordinated time-step progression. Data access is protected
by a list of `Lock` objects (`location_lock`) on a per-location basis.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrier # Assumed to contain ReusableBarrier class.

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, multiple shared barriers, and location-specific locks.
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
        self.thread.start()

        self.barrier1 = None # First shared ReusableBarrier for synchronization.
        self.barrier2 = None # Second shared ReusableBarrier for synchronization.

        self.location_lock = [] # List of Locks, one for each unique data location.

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (two barriers and location-specific locks) among all devices.
        Only the device with `device_id == 0` is responsible for initializing these resources,
        which are then distributed to all other devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: The device with `device_id == 0` initializes the shared barriers.
        if self.device_id == 0:
            self.barrier1 = ReusableBarrier(len(devices))
            self.barrier2 = ReusableBarrier(len(devices))

            # Block Logic: Distributes the initialized shared barriers to all devices.
            for device in devices:
                device.barrier1 = self.barrier1
                device.barrier2 = self.barrier2

            # Block Logic: Determines the maximum location index and initializes a list of Locks
            # for each potential location.
            max_loc = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > max_loc:
                        max_loc = location
            # Inline: Fills the `location_lock` list with `Lock` instances for each location.
            while max_loc >= 0:
                self.location_lock.append(Lock())
                max_loc = max_loc - 1
            
            # Block Logic: Distributes the initialized location-specific locks to all devices.
            for device in devices:
                device.location_lock = self.location_lock


    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        If a script is provided, it's added to the queue. If no script (i.e., `None`)
        is provided, it signals `script_received` and then waits on `barrier2`.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            
        else:
            # Block Logic: If no script is provided (e.g., end of scripts for timepoint),
            # it signals `script_received` and then waits on `barrier2`.
            # This indicates a specific phase of synchronization for script assignment completion.
            self.script_received.set()
            self.barrier2.wait()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `DeviceThread` will acquire the appropriate `location_lock` before calling this method.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `DeviceThread` will acquire the appropriate `location_lock` before calling this method.
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
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts sequentially, and coordinating with other device threads using
    two shared `ReusableBarrier` instances and location-specific locks.
    Time Complexity: O(T * S * (N * D_access + D_script_run)) where T is the number of timepoints,
    S is the number of scripts per device, N is the number of neighbors, D_access is data access
    time, and D_script_run is script execution time.
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
        2. Waits for `script_received` event to be set, and then synchronizes using `barrier2` and `barrier1`.
           Invariant: All device threads are synchronized at multiple stages before script execution.
        3. Executes assigned scripts sequentially: for each script, it acquires the location-specific lock,
           collects data from neighbors and itself, runs the script, updates data on neighbors and itself,
           and then releases the lock.
           Invariant: Data access and modification for a given location are protected by its corresponding lock.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Waits for `script_received` to be set, indicating scripts are assigned.
            # Then, synchronizes with other devices using `barrier2` and `barrier1`, defining timepoint boundaries.
            self.device.script_received.wait()
            self.device.barrier2.wait()
            self.device.barrier1.wait()

            if neighbours is None:
                break

            # Block Logic: Processes each script assigned to the device for the current timepoint.
            # Invariant: Each script retrieves data from neighbors and itself, executes, and updates data,
            # all while holding the appropriate location-specific lock.
            for (script, location) in self.device.scripts:
                # Block Logic: Acquires the lock specific to the data location to ensure exclusive access.
                self.device.location_lock[location].acquire()

                script_data = []
                
                # Block Logic: Collects data from neighboring devices for the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collects data from its own device for the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if any data was collected and propagates the result.
                if script_data != []:
                    
                    result = script.run(script_data)

                    # Block Logic: Updates neighboring devices with the script's result.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Updates its own device's data with the script's result.
                    self.device.set_data(location, result)

                # Block Logic: Releases the location-specific lock after all data operations for this script are complete.
                self.device.location_lock[location].release()
