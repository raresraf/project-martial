"""
This module implements a multi-threaded device simulation framework.

It defines a `Device` class representing simulated entities, a `DeviceThread`
to manage the main operational loop of each device, and `ScriptThread`s
that execute specific data processing scripts. The framework supports
inter-device communication and script processing in a concurrent manner,
with explicit synchronization for accessing shared data locations across
multiple threads and devices. A condition-based reusable barrier
(`ReusableBarrierCond` from `barrier.py`) is used for global synchronization.
"""

from threading import Event, Thread, Lock, Condition
import barrier # Assumed to provide a `ReusableBarrierCond` class.


class Device(object):
    """
    Represents a single simulated device in the system.

    Each device holds its own `sensor_data`, interacts with a `supervisor`,
    and processes scripts. It manages a `DeviceThread` for its main operations
    and coordinates with other devices through a shared barrier (`bariera`) and
    explicit per-location synchronization mechanisms (`locationcondition`, `locationlist`).
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping locations (int) to sensor data values.
            supervisor (object): A reference to the supervisor object managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal when new scripts have been assigned.
        self.scripts = []               # List of (script, location) tuples assigned to this device.
        self.timepoint_done = Event()   # Event to signal that all scripts for a timepoint have been assigned.

        # The main thread responsible for this device's control flow.
        self.thread = DeviceThread(self)
        self.thread.start() # Start the `DeviceThread` immediately upon device creation.
        
        # `bariera` is a ReusableBarrierCond instance for global device synchronization.
        # It is initialized with a placeholder value (1) and will be properly set up by device 0.
        self.bariera = barrier.ReusableBarrierCond(1)
        
        self.data_lock = Lock() # Lock to protect `self.sensor_data` during `get_data` and `set_data` operations.
        
        self.script_lock = Lock() # Lock to protect `self.scripts` list and `script_received` event.
        
        # `locationcondition` is a Condition variable used for coordinating access to data locations.
        # It allows `ScriptThread`s to wait if a location is already being processed.
        self.locationcondition = Condition()
        
        # `locationlist` is a shared list that tracks which data locations are currently locked
        # by a `ScriptThread` to prevent multiple threads from processing the same location concurrently.
        self.locationlist = []

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global shared synchronization primitives (`bariera`, `locationcondition`, `locationlist`).
        This method is designed to be called once by all devices, but global initialization
        logic is handled by the device with `device_id` 0.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Global initialization: Only device with `device_id` 0 creates and distributes
        # the global barrier and location synchronization primitives.
        if self.device_id is 0:
            self.bariera = barrier.ReusableBarrierCond(len(devices)) # Initialize global barrier.
            for device in devices:
                device.bariera = self.bariera                 # Distribute the barrier to all devices.
                device.locationcondition = self.locationcondition # Distribute the shared condition variable.
                device.locationlist = self.locationlist       # Distribute the shared location tracking list.

    def assign_script(self, script, location):
        """
        Assigns a processing script and its associated data location to this device.
        If `script` is None, it signals that script assignment for the current
        timepoint is complete. Access to `self.scripts` is protected by `script_lock`.

        Args:
            script (object): The script object (must have a `run` method) to be executed,
                             or None to signal timepoint completion.
            location (int): The data location this script pertains to.
        """
        self.script_lock.acquire() # Acquire lock to protect `self.scripts` and `script_received`.

        if script is not None:
            self.scripts.append((script, location)) # Add the script to the device's list.
            self.script_received.set()              # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # Signal that all scripts for the current timepoint have been assigned.
        self.script_lock.release() # Release the script lock.

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location within this device.
        Access to `self.sensor_data` is protected by `self.data_lock`.

        Args:
            location (int): The location ID for which to retrieve data.

        Returns:
            Any: The data at the specified location, or None if the location is not found.
        """
        self.data_lock.acquire() # Acquire lock before accessing sensor_data.
        value = self.sensor_data[location] if location in self.sensor_data \
                                           else None
        self.data_lock.release() # Release lock after access.
        return value

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location within this device.
        Access to `self.sensor_data` is protected by `self.data_lock`.

        Args:
            location (int): The location ID for which to set data.
            data (Any): The new data value to be set.
        """
        self.data_lock.acquire() # Acquire lock before modifying sensor_data.
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.data_lock.release() # Release lock after modification.

    def shutdown(self):
        """
        Initiates the graceful shutdown sequence for the device's main thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a `Device`.

    It is responsible for interacting with the supervisor to get neighbor
    information, managing timepoint progression, and dynamically spawning
    `ScriptThread` instances for script execution. It also participates
    in global synchronization using a shared barrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously processes timepoints:
        1. Retrieves neighbor information from the supervisor.
        2. Waits for all scripts for the timepoint to be assigned.
        3. Spawns `ScriptThread` instances for each script.
        4. Waits for all `ScriptThread` instances to complete.
        5. Participates in global barrier synchronization (`device.bariera`).
        """
        while True:
            # Retrieves the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # If no neighbors are returned (e.g., shutdown signal from supervisor),
            # break the loop and terminate this thread.
            if neighbours is None:
                break
            
            # Waits until all scripts for the current timepoint have been assigned.
            # This ensures that `self.device.scripts` is fully populated for the timepoint.
            self.device.timepoint_done.wait()
            
            # Acquire script_lock to safely access `self.device.scripts` and associated state.
            self.device.script_lock.acquire()

            # Creates and manages `ScriptThread` instances for each assigned script.
            nodes = [] # List to hold `ScriptThread` instances.
            for (script, location) in self.device.scripts:
                # Pass shared synchronization primitives (`locationlist`, `locationcondition`) to each `ScriptThread`.
                nodes.append(ScriptThread(self.device, script, location,
                             neighbours, self.device.locationlist,
                             self.device.locationcondition))
            
            # Starts all `ScriptThread` instances.
            for j in xrange(len(self.device.scripts)):
                nodes[j].start()
            
            # Waits for all `ScriptThread` instances to complete their tasks.
            for j in xrange(len(self.device.scripts)):
                nodes[j].join()
            
            # Clears the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()
            
            self.device.script_lock.release() # Release the script lock.
            
            # Participates in the global barrier synchronization, waiting for all devices
            # to complete their current timepoint processing of scripts.
            self.device.bariera.wait()


class ScriptThread(Thread):
    """
    A worker thread responsible for executing a single assigned script task.

    Instances of `ScriptThread` are created and managed by `DeviceThread`.
    It uses a shared `Condition` variable and a list to synchronize access
    to specific data locations.
    """
    
    def __init__(self, device, script, location, neighbours, locationlist,
                 locationcondition):
        """
        Initializes a ScriptThread instance.

        Args:
            device (Device): The Device instance this worker operates for.
            script (object): The script object (must have a `run` method) to be executed.
            location (int): The data location relevant to this script.
            neighbours (list): The list of neighboring devices for the current timepoint.
            locationlist (list): A shared list tracking currently locked locations.
            locationcondition (Condition): A shared Condition variable for coordinating location access.
        """
        Thread.__init__(self, name="Script Thread for Device %d, Location %d" % (device.device_id, location))
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.locationlist = locationlist
        self.locationcondition = locationcondition

    def run(self):
        """
        The main execution logic for `ScriptThread`.

        It acquires an exclusive lock for its assigned data `location` using
        the shared `locationcondition` and `locationlist`. It then collects
        data, executes the script, updates data in relevant devices, and
        finally releases the location lock, notifying other waiting threads.
        """
        sem = 1 # Flag to control the while loop for acquiring location lock.
        
        # Block Logic: Acquire exclusive access to the `self.location`.
        # Uses a Condition variable to wait if the location is already being processed
        # by another `ScriptThread` (indicated by its presence in `locationlist`).
        while sem is 1:
            self.locationcondition.acquire() # Acquire the lock associated with the Condition.
            if self.location in self.locationlist:
                # If location is already locked, wait until it's released.
                self.locationcondition.wait()
            else:
                # If location is free, acquire it by adding to `locationlist` and proceed.
                self.locationlist.append(self.location)
                sem = 0 # Exit loop.
            self.locationcondition.release() # Release the condition lock.

        # Collect data from all neighboring devices for the current location.
        script_data = []
        for device in self.neighbours:
            # `get_data` is protected by device's `data_lock`.
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            # The next two lines seem to be duplicate calls for `self.device.get_data`.
            # This is likely a copy-paste error or intended for a different structure.
            # Only the first `get_data` call for `self.device` should be kept.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # If any data was collected, run the script and update devices.
        if script_data != []:
            result = self.script.run(script_data) # Execute the script.
            
            # Updates the data in neighboring devices with the script's result.
            # `set_data` is protected by device's `data_lock`.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Updates its own device's data with the script's result.
            # `set_data` is protected by device's `data_lock`.
            self.device.set_data(self.location, result)
        
        # Block Logic: Release exclusive access to the `self.location`.
        # Removes the location from the shared list and notifies any waiting threads.
        self.locationcondition.acquire() # Re-acquire the condition lock.
        self.locationlist.remove(self.location) # Remove location from the locked list.
        self.locationcondition.notify_all() # Notify all waiting threads that a location has been freed.
        self.locationcondition.release() # Release the condition lock.