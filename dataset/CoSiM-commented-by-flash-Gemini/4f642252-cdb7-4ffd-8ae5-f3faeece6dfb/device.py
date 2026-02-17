"""
@4f642252-cdb7-4ffd-8ae5-f3faeece6dfb/device.py
@brief Implements a simulated device in a distributed sensor network with concurrent script execution and custom synchronization.
This module defines a `Device` that processes sensor data, interacts with neighbors,
and executes scripts. It features a `DeviceThread` that dispatches scripts to
`ScriptThread` instances, using semaphores and events for signaling, and `RLock` objects
for per-location data protection across devices. A custom barrier-like synchronization
mechanism is also implemented.
"""

from threading import Event, Thread, Semaphore, current_thread, RLock


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, custom synchronization primitives, and
    location-specific reentrant locks.
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
        self.script_received = Semaphore(0) # Semaphore to signal that new scripts are available for processing.
        self.scripts = [] # List to store assigned scripts.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.all_devices = None # Will store a reference to all devices in the simulation.
        self.timepoint_sem = Semaphore(0) # Semaphore used for timepoint synchronization among devices.
        self.list_thread = [] # List to keep track of active `ScriptThread` instances.
        self.loc_lock = {} # Dictionary to hold RLock objects for each data location.

    def __str__(self):
        """
        @brief Provides a string representation of the device, including the current thread.
        @return A formatted string showing the current thread and device ID.
        """
        return "[%.35s]    Device %d:" % (current_thread(),self.device_id)

    def sync_on_timepoint(self):
        """
        @brief Implements a custom synchronization barrier for timepoints.
        Blocks the calling thread until all other devices have signaled their completion
        for the current timepoint by releasing their `timepoint_sem`.
        Invariant: All devices wait here until `len(self.all_devices) - 1` releases are acquired.
        """
        for i in range(len(self.all_devices)-1):
            self.timepoint_sem.acquire()

    def setup_devices(self, devices):
        """
        @brief Sets up the list of all devices in the simulation.
        @param devices: A list of all Device instances in the simulation.
        """
        self.all_devices = devices

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Initializes an `RLock` for the given `location` if one doesn't already exist.
        Signals that a script has been received by releasing the `script_received` semaphore.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Block Logic: Initializes an RLock for the specific location if it doesn't exist.
            # RLock allows the same thread to acquire it multiple times.
            if location not in self.loc_lock:
                self.loc_lock[location] = RLock()
        else:
            # Block Logic: Signals completion of the timepoint if no script is assigned.
            self.timepoint_done.set()
        # Block Logic: Releases the semaphore to indicate that a script has been assigned.
        self.script_received.release()

    def lock(self, location):
        """
        @brief Acquires the RLock for a given data location.
        If the lock does not exist for the location, it is created.
        @param location: The data location for which to acquire the lock.
        """
        if not location in self.loc_lock:
            self.loc_lock[location] = RLock()
        self.loc_lock[location].acquire()

    def unlock(self, location):
        """
        @brief Releases the RLock for a given data location.
        @param location: The data location for which to release the lock.
        """
        try:
            self.loc_lock[location].release()
        except RuntimeError:
            # Inline: Handles cases where the lock is released without being acquired by the current thread,
            # which can happen with RLock if release is called more times than acquire.
            pass


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

class ScriptThread(Thread):
    """
    @brief A dedicated thread for executing a single script for a specific data location.
    This thread is responsible for gathering data, running the script, and then
    propagating the results to relevant devices, ensuring thread-safe access to data
    through location-specific RLocks.
    """
    
    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes a `ScriptThread` instance.
        @param device: The parent `Device` instance for which the script is being run.
        @param location: The data location that the script operates on.
        @param script: The script object to execute.
        @param neighbours: A list of neighboring `Device` instances.
        """
        Thread.__init__(self)


        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for `ScriptThread`.
        Block Logic:
        1. Acquires locks on neighboring devices for the specified `location` to prevent race conditions during data access.
        2. Collects data from neighboring devices and its own device for the specified `location`.
        3. Executes the assigned `script` if any data was collected.
        4. Propagates the script's `result` to neighboring devices and its own device.
        5. Releases the locks on neighboring devices.
        Invariant: All data access and modification for a given `location` involving neighbors are protected by RLocks.
        """
        script_data = []
        
        # Block Logic: Acquires RLocks on all neighboring devices for the specific location.
        # This prevents other threads from modifying this location's data on neighbors during script execution.
        if self.neighbours:
            for device in self.neighbours:
                device.lock(self.location)
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
            if self.neighbours:
                for device in self.neighbours:
                    device.set_data(self.location, result)

            # Block Logic: Acquires the RLock for its own device's location, updates data, and then releases the lock.
            self.device.lock(self.location)
            self.device.set_data(self.location, result)
            self.device.unlock(self.location)

        # Block Logic: Releases RLocks on all neighboring devices after data propagation is complete.
        if self.neighbours:
            for device in self.neighbours:
                device.unlock(self.location)



class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts concurrently via `ScriptThread` instances, and coordinating with
    other device threads using custom semaphore-based timepoint synchronization.
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
        2. Waits for the `timepoint_done` event to be set (signals readiness for timepoint processing).
        3. Acquires the `script_received` semaphore (signals that scripts are assigned).
        4. Creates and starts a `ScriptThread` for each assigned script, storing them in `list_thread`.
           Invariant: All scripts for the current timepoint are executed concurrently.
        5. Waits for all `ScriptThread` instances to complete.
        6. Signals other devices about its timepoint completion by releasing their `timepoint_sem`.
        7. Clears the `timepoint_done` event for the next cycle.
        8. Waits for other devices to complete their timepoint processing using `sync_on_timepoint`.
           Invariant: All devices globally synchronize at the end of each timepoint.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the device's timepoint is marked as done, allowing script processing to begin.
            self.device.timepoint_done.wait()
            # Block Logic: Acquires the semaphore, waiting until a script has been assigned to this device.
            self.device.script_received.acquire()

            # Block Logic: Iterates through assigned scripts, creating and starting a new `ScriptThread` for each.
            # Invariant: Each script is executed concurrently in its own `ScriptThread`.
            for (script, location) in self.device.scripts:
                thread = ScriptThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)
                thread.start()

            # Block Logic: Waits for all `ScriptThread` instances to complete their execution.
            for t in self.device.list_thread:
                t.join()
            self.device.list_thread = [] # Clears the list of threads.

            # Block Logic: Signals all other devices that this device has completed its timepoint processing.
            for d in self.device.all_devices:
                if d == self.device:
                    continue
                d.timepoint_sem.release()

            # Block Logic: Clears the `timepoint_done` event for the next timepoint cycle.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes with all other devices to ensure all have completed the current timepoint.
            self.device.sync_on_timepoint()
