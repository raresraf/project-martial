


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `Device`: Represents a single device, managing its sensor data and orchestrating operations.
- `MyThread`: A worker thread responsible for executing individual scripts.
- `DeviceThread`: The main orchestrating thread for a `Device`, which dynamically
  creates and manages `MyThread`s to execute assigned scripts in batches.

The system utilizes `ReusableBarrierSem` from the `barrier` module for inter-device
synchronization, `threading.Event` for signaling (e.g., `script_received`, `ready`),
and `threading.Lock` for protecting data access (e.g., `get_data_lock`, and a shared
list of `locations` locks).
"""


from threading import Event, Thread, Lock
import supervisor
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution through its `DeviceThread`.
    It uses shared locks for data locations and a barrier for synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for the device.
            supervisor (object): A reference to a supervisor object for inter-device communication.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are assigned.
        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).
        self.timepoint_done = Event() # Event to signal that the current timepoint's processing is complete (appears unused here).
        self.locations = [] # List to hold shared locks for specific data locations.
        self.get_data_lock = Lock() # A lock to protect access to `sensor_data` during `get_data` calls.
        self.ready = Event() # Event to signal that this device's setup is complete.
        self.devices = None # Placeholder for a list of all devices in the simulation, set in `setup_devices`.
        self.barrier = None # Placeholder for the global ReusableBarrierSem, set in `setup_devices`.
        self.thread = DeviceThread(self) # The main orchestrating thread for this device.
        self.thread.start() # Start the DeviceThread.

    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the global barrier and shared location-specific locks,
        then distributes them to all devices. This method is designed to be
        called only by the device with `device_id == 0`. It creates a single
        `ReusableBarrierSem` and a list of `Lock`s for each data location
        (up to 150 assumed locations), then distributes these to all devices
        and signals their readiness.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        self.devices = devices # Stores references to all devices.
        # Inline: Creates a global `ReusableBarrierSem` for synchronization among all DeviceThreads.
        barrier = ReusableBarrierSem(len(devices))
        
        # Block Logic: Only the device with device_id 0 initializes the shared `locations` list of locks.
        if self.device_id == 0:
            i = 0
            while i < 150:  # Assuming a maximum of 150 distinct data locations.
                self.locations.append(Lock()) # Appends a new `Lock` for each location.
                i = i + 1

            # Block Logic: Distributes the created global barrier and location locks to all devices.
            for dev in devices:
                dev.barrier = barrier
                dev.locations = self.locations # All devices share the same list of locks.
                dev.ready.set() # Signals that this device's setup is complete.

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals that script assignments are complete.
        If a script is provided, it's appended to the device's internal script list.
        If `script` is None, it signals that script assignments for the current
        timepoint are complete by setting `script_received`.

        Args:
            script (object): The script object to be executed, or None to signal end of assignments.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
        else:
            self.script_received.set() # If script is None, signal that script assignments are done for the timepoint.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's `sensor_data` dictionary.
        Access to `sensor_data` is protected by `get_data_lock`.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        with self.get_data_lock: # Acquire lock to protect concurrent access to sensor_data.
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location in this device's `sensor_data` dictionary.
        The data is updated only if the location exists in the `sensor_data`.

        Args:
            location (int): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by waiting for its main `DeviceThread` to complete.
        """
        self.thread.join() # Wait for the DeviceThread to finish its execution.


class DeviceThread(Thread):
    """
    The main orchestrating thread for a `Device`.
    This thread is responsible for fetching neighbor information from the supervisor,
    managing script execution by dynamically creating and managing `MyThread`s in batches,
    and participating in global synchronization.
    """

    def __init__(self, device):
        """
        Initializes a `DeviceThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
        """
        
        Thread.__init__(self, name="%d" % device.device_id) # Set thread name to device ID.
        self.device = device

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        It waits for the device to be ready, continuously fetches neighbor data,
        manages script execution by dynamically creating and managing `MyThread`s
        in batches, and participates in global synchronization.
        """
        self.device.ready.wait() # Block Logic: Wait until the parent device's setup is complete.

        while True:
            neigh = self.device.supervisor.get_neighbours() # Inline: Fetch neighbor information from the supervisor.
            if neigh is None: # Inline: If `neigh` is None, it signals termination for the device.
                break # Exit the main loop, terminating the DeviceThread.

            # Block Logic: Wait for scripts to be assigned for the current timepoint and then clear the event.
            self.device.script_received.wait()
            self.device.script_received.clear()

            rem_scripts = len(self.device.scripts) # Get the total number of scripts assigned for this timepoint.

            threads = [] # List to hold `MyThread` instances for current script batch.
            i = 0
            # Inline: Create a `MyThread` for each assigned script.
            while i < rem_scripts:
                threads.append(MyThread(self.device, neigh, self.device.scripts, i))
                i = i + 1

            # Block Logic: Dynamic batching strategy for starting and joining `MyThread`s.
            # This aims to limit the number of concurrently running threads.
            if rem_scripts < 8: # If fewer than 8 scripts, start and join all at once.
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            else: # If 8 or more scripts, process them in batches of 8.
                pos = 0 # Starting index for the current batch.
                while rem_scripts != 0: # Continue until all scripts are processed.
                    if rem_scripts > 8: # If more than 8 scripts remaining, process a batch of 8.
                        for i in range(pos, pos + 8):
                            threads[i].start()
                        for i in range(pos, pos + 8):
                            threads[i].join()
                        pos = pos + 8 # Move to the next batch starting position.
                        rem_scripts = rem_scripts - 8 # Decrement remaining scripts.
                    else: # If 8 or fewer scripts remaining, process the last batch.
                        for i in range(pos, pos + rem_scripts):
                            threads[i].start()
                        for i in range(pos, pos + rem_scripts):
                            threads[i].join()
                        rem_scripts = 0 # All scripts processed.

            # Block Logic: Synchronize with other devices at the global barrier.
            # This ensures all devices have completed their script processing for the timepoint.
            self.device.barrier.wait()


class MyThread(Thread):
    """
    A worker thread responsible for executing a single script from the `Device`'s script list.
    It collects necessary data from neighbors and its parent device, runs the assigned script,
    and updates the sensor data on relevant devices, all while acquiring a location-specific lock.
    """

    def __init__(self, device, neigh, scripts, index):
        """
        Initializes a `MyThread` instance.

        Args:
            device (Device): The parent `Device` object that spawned this thread.
            neigh (list): A list of Device objects representing neighboring devices.
            scripts (list): The list of all scripts assigned to the parent `Device`.
            index (int): The index of the specific script in `scripts` that this thread should execute.
        """
        
        Thread.__init__(self, name="%d" % device.device_id) # Set thread name to device ID.
        self.device = device
        self.neigh = neigh # List of neighboring devices.
        self.scripts = scripts # Reference to the full list of device scripts.
        self.index = index # Index of the script to execute.

    def run(self):
        """
        The main execution method for the `MyThread`.
        It retrieves its assigned script, acquires the location-specific lock,
        collects data from neighbors and its parent device, executes the script,
        and then updates the sensor data on relevant devices with the script's result,
        before releasing the lock.
        """
        # Block Logic: Unpack the script and location for this worker from the shared scripts list.
        (script, loc) = self.scripts[self.index]
        
        # Block Logic: Acquire the location-specific lock to ensure exclusive access
        # to data at this `loc` across all devices during script execution and data update.
        self.device.locations[loc].acquire()
        
        info = [] # List to collect input data for the script.
        # Block Logic: Collect data from all neighboring devices at the specified location.
        for neigh_iter in self.neigh: # `neigh_iter` is a variable name for a neighbor device.
            aux_data = neigh_iter.get_data(loc) # Get data from the neighbor.
            if aux_data is not None:
                info.append(aux_data) # Add to script input if available.
        
        # Block Logic: Collect data from this worker's own parent device at the specified location.
        aux_data = self.device.get_data(loc)
        if aux_data is not None:
            info.append(aux_data) # Add to script input if available.
        
        # Block Logic: If input data is available, execute the script and update device data.
        if info != []:
            result = script.run(info) # Inline: Execute the script's `run` method with the collected data.
            
            # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.
            for neigh_iter in self.neigh:
                neigh_iter.set_data(loc, result) # Update neighbor's data.
                self.device.set_data(loc, result) # Update this device's own data.
        self.device.locations[loc].release() # Release the location-specific lock.
