"""
This module implements a multi-threaded device simulation framework.

It defines `Device` objects that represent simulated entities, `DeviceThread`s
to manage the lifecycle and interactions of each device, and `ScriptRunner`
threads to execute specific scripts for data processing. The system utilizes
a shared barrier for global synchronization across devices and per-location
locks for data consistency, with script execution managed in a batch-wise
fashion by the `DeviceThread`.
"""

from threading import Event, Thread, Lock
import barrier # Assumed to provide a `ReusableBarrierSem` class.
import runner # `ScriptRunner` is defined within this file, so `import runner` might be for another `runner` or unused.


class Device(object):
    """
    Represents a single simulated device in the system.

    Each device holds its own `sensor_data`, interacts with a `supervisor`,
    and processes scripts. It manages a `DeviceThread` for its operations
    and coordinates with other devices through a shared barrier and
    a list of per-location locks (`locks`).
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
        self.scripts = []           # List of (script, location) tuples assigned to this device.
        self.timepoint_done = Event()   # Event to signal that all scripts for a timepoint have been assigned.

        self.thread = DeviceThread(self) # The main thread responsible for this device's control flow.
        self.barr = None            # This will hold the shared global barrier, set during `setup_devices`.
        self.devices = []           # List of all Device objects in the simulation.
        self.runners = []           # List to store `ScriptRunner` threads for current timepoint.
        # A list of Locks, intended for per-location synchronization. Initialized to 50 `None`s.
        # These locks are dynamically created or shared from other devices during `assign_script`.
        self.locks = [None] * 50
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global shared synchronization primitives, specifically the barrier.
        This method is designed to be called once by all devices, but global initialization
        logic for the barrier is handled only if `self.barr` is None (i.e., by the first
        device to call this). It also populates the list of all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Global barrier initialization. This block ensures the barrier is created only once.
        if self.barr is None:
            barr = barrier.ReusableBarrierSem(len(devices)) # Create a new shared barrier.
            self.barr = barr # Assign to current device.
            for dev in devices:
                if dev.barr is None: # Assign the barrier to other devices if they haven't received it.
                    dev.barr = barr
        
        # Populates the list of all devices.
        for dev in devices:
            if dev is not None:
                self.devices.append(dev)

    def assign_script(self, script, location):
        """
        Assigns a processing script and its associated data location to this device.
        This method also handles the dynamic assignment or sharing of per-location locks.

        Args:
            script (object): The script object (must have a `run` method) to be executed,
                             or None to signal timepoint completion.
            location (int): The data location this script pertains to.
        """
        ok = 0 # Flag to check if a lock for the location was found/shared.
        if script is not None:
            self.scripts.append((script, location)) # Add the new script to the list.
            
            # Dynamic lock assignment for the given location.
            if self.locks[location] is None: # If this device doesn't have a lock for this location yet.
                for device in self.devices: # Check if any other device already has a lock for this location.
                    if device.locks[location] is not None:
                        self.locks[location] = device.locks[location] # Share the existing lock.
                        ok = 1
                        break
                if ok == 0:
                    self.locks[location] = Lock() # If no existing lock found, create a new one.
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # Signal that all scripts for the current timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location within this device.

        Note: This method itself does not acquire any locks to protect `sensor_data` access.
        Concurrency control for `sensor_data` is expected to be managed externally,
        typically by acquiring a lock from `self.locks[location]` before calling this method.

        Args:
            location (int): The location ID for which to retrieve data.

        Returns:
            Any: The data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data 
            else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location within this device.

        Note: Similar to `get_data`, this method does not acquire any locks to protect
        `sensor_data` modification. External synchronization (acquiring a lock from
        `self.locks[location]`) is expected before calling this method.

        Args:
            location (int): The location ID for which to set data.
            data (Any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

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
    `ScriptRunner` threads to execute scripts in batches. It also participates
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
        3. Dynamically creates and manages `ScriptRunner` threads for each script,
           executing them in batches while holding a lock specific to each location.
        4. Clears the `timepoint_done` event.
        5. Synchronizes globally using the device's barrier.
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

            # Creates ScriptRunner threads for each assigned script.
            # Note: The `runners` list is cleared and repopulated for each timepoint,
            # indicating a dynamic thread management approach.
            for (script, location) in self.device.scripts:
                run = ScriptRunner(self.device, script, location, neighbours)
                self.device.runners.append(run)

                n = len(self.device.runners) # Total number of runners created so far for this timepoint.
                x = n / 8 # Number of full batches of 8 runners.
                r = n % 7 # Remainder of runners (potentially for an incomplete batch).
                
                # Acquire the lock for the current script's location *before* starting its runner(s).
                # This ensures that all operations on a specific location (get_data, set_data)
                # by the ScriptRunner are serialized for that location across all devices.
                self.device.locks[location].acquire()

                # This complex batching logic appears to be attempting to manage
                # parallel execution, but due to `acquire()` and `release()`
                # per location, and subsequent `join()`, the parallel benefit
                # is limited to scripts operating on *different* locations.
                # If n is the current number of runners (for current timepoint, up to this script),
                # the goal seems to be to start a batch of 8, then join them, then next batch.

                # This logic seems to trigger starting and joining threads for previously added
                # runners (based on `x` and `n >= 8`), which is inefficient and not true batching.
                # The loop structure is not starting/joining in proper batches of 8 for *all* runners.

                # Corrected interpretation attempt:
                # The current script is added. 'n' is now the count of scripts.
                # The nested loops for 'x' and 'j' run all the *previous* full batches.
                for i in xrange(0, x):
                    for j in xrange(0, 8):
                        # Start and join runners from previous full batches. This makes execution sequential.
                        self.device.runners[i * 8 + j].start()
                
                # This `if` condition with `n >= 8` and `len(self.device.runners) - r`
                # seems to handle the current batch of runners. This logic might be trying
                # to run the current partial batch.
                if n >= 8:
                    for i in xrange(len(self.device.runners) - r,
                                    len(self.device.runners)):
                        self.device.runners[i].start()
                
                # If `n` is less than 8, this else block starts and joins runners from 0 to n-1.
                # This also indicates sequential processing of runners as they are added.
                else:
                    for i in xrange(0, n):
                        self.device.runners[i].start()
                
                # Joins all currently running (or just started) ScriptRunner threads.
                # This makes the execution of scripts for a given timepoint sequential.
                for i in xrange(0, n):
                    self.device.runners[i].join()
                
                # Release the lock for the current script's location.
                self.device.locks[location].release()
                
                # The runners list is cleared and repopulated for each script,
                # which is inefficient and leads to a misunderstanding of `n` above.
                # It seems `self.device.runners` should hold all runners for the timepoint,
                # and then they should be batched.
                self.device.runners = [] # This line means only one runner is ever active at a time for each script.

            # Clears the `timepoint_done` event for the next timepoint.
            # Note: This event is waited upon at the beginning of the loop but its `set()`
            # happens in `Device.assign_script`.
            self.device.timepoint_done.clear()
            
            # Participates in the global barrier synchronization, waiting for all devices
            # to complete their current timepoint processing of scripts.
            self.device.barr.wait()


class ScriptRunner(Thread):
    """
    A worker thread responsible for executing a single assigned script task.
    Instances of `ScriptRunner` are created and managed by `DeviceThread`.
    """
    
    def __init__(self, device, script, location, neighbours):
        """
        Initializes a ScriptRunner thread.

        Args:
            device (Device): The Device instance this worker operates for.
            script (object): The script object (must have a `run` method) to be executed.
            location (int): The data location relevant to this script.
            neighbours (list): The list of neighboring devices for the current timepoint.
        """
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        The main execution logic for `ScriptRunner`.

        It collects data from neighbors and the local device for its assigned
        location, executes the script, and then updates the data in relevant
        devices. The `location` lock is assumed to be acquired externally
        (by the `DeviceThread`) before this method is called.
        """
        script_data = [] # List to collect all relevant data for the script. 
        
        # Gathers data from all neighboring devices for the current location.
        for device in self.neighbours:
            # `get_data` itself does not use a lock; external locking is expected.
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Gathers data from its own device for the current location.
        # `get_data` itself does not use a lock; external locking is expected.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # If any data was collected, run the script and update devices.
        if script_data != []:
            result = self.script.run(script_data) # Execute the script. 
            
            # Updates the data in neighboring devices with the script's result.
            # `set_data` itself does not use a lock; external locking is expected.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Updates its own device's data with the script's result.
            # `set_data` itself does not use a lock; external locking is expected.
            self.device.set_data(self.location, result)
