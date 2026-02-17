"""
@6008acd3-3e51-4953-832a-e356e97fdc71/device.py
@brief Implements a simulated device for a distributed sensor network, with sequential script execution and shared synchronization primitives.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` for operational logic and uses a shared `ReusableBarrier`
for global time-step synchronization and a shared `Lock` for data access protection.
"""

from threading import Event, Thread, Lock
import barrier # Assumed to contain ReusableBarrier class.

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, a shared barrier, and a shared lock for data consistency.
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
        self.my_lock = None # Shared lock for data access, to be initialized by device 0.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.bariera = None # Shared barrier for global time step synchronization, to be initialized by device 0.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared `ReusableBarrier` and `Lock` for synchronization among all devices.
        Only the device with `device_id == 0` initializes these shared resources,
        which are then distributed to all other devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: Initializes shared synchronization primitives if this is the first device.
        # Invariant: A single `ReusableBarrier` and `Lock` instance are created and shared across all devices.
        if self.device_id == 0:
            lent = len(devices)
            bariera = barrier.ReusableBarrier(lent) # Creates the shared barrier.

            my_lock = Lock() # Creates the shared lock.
            # Block Logic: Distributes the shared barrier to all devices.
            for device in devices:
                device.bariera = bariera
            # Block Logic: Distributes the shared lock to all devices.
            for device in devices:
                device.my_lock = my_lock

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Signals that a script has been received, or that a timepoint is done if no script.
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
        Note: This method does not acquire `self.my_lock` directly; it is assumed that external
        mechanisms (e.g., in `DeviceThread`) will handle the locking for data access.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire `self.my_lock` directly; it is assumed that external
        mechanisms (e.g., in `DeviceThread`) will handle the locking for data modification.
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
    a shared `ReusableBarrier` and a shared `Lock` for data access.
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
        2. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.
        3. Acquires the shared `self.device.my_lock` to protect data access during script execution.
        4. Processes each assigned script: it collects data from neighbors and itself,
           runs the script, and then updates data on neighbors and itself.
           Invariant: All data access and modification for a given location are protected by `self.device.my_lock`.
        5. Releases the shared `self.device.my_lock`.
        6. Clears the `timepoint_done` event for the next cycle.
        7. Synchronizes with all other device threads using a shared `ReusableBarrier`.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break


            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()
            
            # Block Logic: Acquires the shared lock to protect data access during script execution for this device.
            self.device.my_lock.acquire()
            # Block Logic: Processes each script assigned to the device for the current timepoint.
            # Invariant: Each script retrieves data from neighbors and itself, executes, and updates data,
            # all while holding the shared lock.
            for (script, location) in self.device.scripts:
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
                if script_data != []:
                    
                    result = script.run(script_data)
		    
		    
                    # Block Logic: Updates neighboring devices with the script's result.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Updates its own device's data with the script's result.
                    self.device.set_data(location, result)
                # Inline: Note the release of the lock for each script. This makes data access safe for the current script.
            self.device.my_lock.release()
	    
            # Block Logic: Clears the `timepoint_done` event for the next timepoint cycle.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.bariera.wait()
