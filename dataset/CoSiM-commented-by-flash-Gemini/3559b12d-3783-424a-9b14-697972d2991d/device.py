"""
@05314305-b286-4c5b-a80e-5c46defa6a97/arch/arm/crypto/Makefile device.py
@brief Implements a device-centric distributed processing system for managing sensor data, executing scripts, and coordinating operations across a network of devices.
This module defines the `Device` class, which encapsulates device-specific data and control logic,
`DeviceThreadHelper` for executing individual scripts in parallel, and `DeviceThread` for orchestrating
the device's main operational cycle including synchronization with other devices using a reusable barrier.
It also manages fine-grained locking for data locations and script execution.
"""


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    """
    @brief Represents a single device in the distributed processing network.
    Each device manages its own sensor data, executes assigned scripts, and coordinates
    with a central supervisor and other devices for synchronized operations.
    It encapsulates synchronization primitives like location-specific locks and a barrier.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id A unique identifier for this device.
        @param sensor_data A dictionary or similar structure storing sensor data, keyed by location.
        @param supervisor The supervisor object responsible for managing the overall device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to hold scripts assigned to this device for execution.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's script assignments.
        self.thread = DeviceThread(self) # The main thread responsible for the device's operational cycle.

        self.location_locks = {} # Dictionary of Lock objects, one for each data location, for fine-grained access control.
        self.script_locks = {} # Dictionary of Lock objects, one for each script, to prevent concurrent execution of the same script logic.
        self.barrier = None # Reusable barrier for synchronizing all devices.

        self.threads = [] # List to hold references to helper threads executing scripts.
        self.thread.start() # Starts the main device thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A formatted string "Device [device_id]".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared reusable barrier for all devices in the network.
        This function is typically called once by a coordinating entity to ensure
        all devices share the same barrier instance for synchronization.
        @param devices A list of all `Device` objects in the network.
        """
        # Block Logic: Create a single reusable barrier for all devices.
        barrier = ReusableBarrierSem(len(devices))
        # Block Logic: Assign the newly created barrier to each device in the network.
        for device in devices:
            device.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location on this device.
        If `script` is `None`, it signals that all scripts for the current timepoint
        have been assigned, and the `timepoint_done` event is set.
        @param script The script (callable) to assign, or `None` to signal timepoint completion.
        @param location The data location associated with the script.
        """
        # Block Logic: If a script is provided, add it to the list of scripts to be executed.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If `script` is None, it acts as a sentinel to signal the end of script assignment for a timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location on this device.
        @param location The identifier of the data location.
        @return The sensor data corresponding to the `location`, or `None` if the location is not found.
        """
        # Inline: Safely access `sensor_data` dictionary, returning `None` if the key doesn't exist.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location on this device.
        The data is only updated if the `location` already exists in the `sensor_data`.
        @param location The identifier of the data location.
        @param data The new data to be set for the specified location.
        """
        # Block Logic: Update data only if the location key already exists in the sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device.
        This involves waiting for the main device thread and all associated helper threads
        (that were created for script execution) to complete their execution.
        """
        self.thread.join() # Wait for the main device thread to finish.
        # Block Logic: Wait for any currently running helper threads (script execution threads) to complete.
        for thread in self.threads:
            thread.join()


class DeviceThreadHelper(Thread):
    """
    @brief A helper thread responsible for executing a single script at a specific location.
    This thread handles data retrieval from neighboring devices, acquires necessary locks,
    executes the provided script, and updates data across the network.
    """
    

    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes a DeviceThreadHelper instance.
        @param device The parent `Device` instance.
        @param script The script (callable) to be executed by this helper thread.
        @param location The data location associated with this script.
        @param neighbours A list of neighboring devices from which to fetch or update data.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script


    def run(self):
        """
        @brief The main execution logic for the helper thread.
        This method ensures synchronized access to data locations and scripts,
        gathers data, executes the script, and propagates results.
        """
        script_data = []

        # Block Logic: Ensure a lock exists for the current data location.
        if self.location not in self.device.location_locks:
            self.device.location_locks[self.location] = Lock()

        # Block Logic: Ensure a lock exists for the current script.
        if self.script not in self.device.script_locks:
            self.device.script_locks[self.script] = Lock()

        self.device.location_locks[self.location].acquire() # Acquire lock for the data location.

        # Block Logic: Gather data from neighboring devices and the current device.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Block Logic: If data is available, execute the script and update results.
        if script_data != []:
            self.device.script_locks[self.script].acquire() # Acquire lock for the script to prevent concurrent execution of the same script logic.
            result = self.script.run(script_data) # Execute the script with collected data.
            self.device.script_locks[self.script].release() # Release script lock.

            # Block Logic: Propagate the results to neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)

        self.device.location_locks[self.location].release() # Release lock for the data location.


class DeviceThread(Thread):
    """
    @brief The main operational thread for a Device.
    This thread orchestrates the device's behavior over time, including fetching
    neighbor information, dispatching scripts to helper threads for execution,
    and synchronizing with other devices using a shared barrier.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.
        @param device The `Device` instance that this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device's operational cycle.
        This loop continuously fetches neighbor information, dispatches scripts
        (if available) to helper threads, waits for their completion, and
        synchronizes with other devices at the end of each timepoint.
        """
        while True:
            # Block Logic: Fetch information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If `neighbours` is None, it signals that the simulation should terminate.
            if neighbours is None:
                break # Exit the main loop to terminate the thread.


            # Invariant: The `neighbours` list always includes the current device itself for local data access.
            neighbours.append(self.device) # Add the current device to the list of neighbors for data processing.

            # Block Logic: Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait() # Blocks until `timepoint_done` event is set.

            # Block Logic: Iterate through assigned scripts and launch helper threads for each.
            for (script, location) in self.device.scripts:
                # Create a helper thread to execute the script for the given location and neighbors.
                thread = DeviceThreadHelper(self.device, script, location, neighbours)
                self.device.threads.append(thread) # Keep a reference to the helper thread.
                thread.start() # Start the helper thread.

            # Block Logic: Wait for all helper threads launched in this timepoint to complete.
            for thread in self.device.threads:
                thread.join() # Blocks until each helper thread finishes execution.

            self.device.threads = [] # Clear the list of helper threads for the next timepoint.

            # Block Logic: Synchronize with all other devices using the shared barrier.
            # This ensures all devices complete their script execution for the current timepoint
            # before proceeding to the next timepoint.
            self.device.barrier.wait()
            # Block Logic: Clear the `timepoint_done` event, resetting it for the next timepoint's script assignment phase.
            self.device.timepoint_done.clear()
