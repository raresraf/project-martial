


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `Device`: Represents a single device, managing its sensor data and orchestrating operations.
- `DeviceThread`: The main orchestrating thread for a `Device`, which spawns `SingleDeviceThread`s
  to execute assigned scripts in batches.
- `SingleDeviceThread`: A worker thread that executes a single script, collects data,
  and updates device states using location-specific locks.
- `ReusableBarrierSem`: A reusable barrier for synchronizing multiple threads in phases.

The system features complex synchronization mechanisms, including a unique setup for shared
`map_locations` locks initialized based on unique locations across devices. It uses
`threading.Event` for signaling and `threading.Lock` for protecting shared data.
"""


from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution through its `DeviceThread`.
    It uses shared `map_locations` locks for data consistency and a global barrier.
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
        self.timepoint_done = Event() # Event to signal that the current timepoint's processing is complete.
        self.thread = DeviceThread(self) # The main orchestrating thread for this device.
        self.thread.start() # Start the DeviceThread.
        self.barrier = None # Placeholder for the global ReusableBarrierSem, set in setup_devices.
        self.map_locations = None # Placeholder for a dictionary of location locks, set in setup_devices.

    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the global barrier and a shared dictionary of location-specific locks,
        then distributes them to all devices. This complex setup is designed to be executed
        only by the device with the lowest `device_id` among all devices to avoid race conditions.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        
        flag = True # Flag to determine if this device has the lowest ID.
        device_number = len(devices) # Total number of devices.

        # Block Logic: Check if this device has the lowest `device_id` among all devices.
        # This determines which device is responsible for initializing global resources.
        for dev in devices:
            if self.device_id > dev.device_id: # If another device has a lower ID.
                flag = False # This is not the lowest ID device.

        if flag == True: # Only the device with the lowest ID performs this setup.
            # Inline: Creates a global `ReusableBarrierSem` for synchronization among all DeviceThreads.
            barrier = ReusableBarrierSem(device_number)
            map_locations = {} # Dictionary to store shared locks for each data location.
            tmp = {} # Temporary dictionary to track unique locations encountered.
            # Block Logic: Identify all unique data locations across all devices and create a `Lock` for each.
            for dev in devices:
                # `tmp` will contain locations from the current `dev.sensor_data` that haven't been mapped yet.
                tmp = list(set(dev.sensor_data) - set(map_locations))
                for i in tmp:
                    map_locations[i] = Lock() # Assign a new Lock for each new unique location.
                dev.map_locations = map_locations # Assign the shared dictionary of locks to this device.
                tmp = {} # Reset temporary dictionary.

            # Block Logic: Distributes the created global barrier and location locks to all devices.
            for dev in devices:
                dev.barrier = barrier
                tmp = list(set(dev.sensor_data) - set(map_locations)) # This line seems to be a duplicate or intended differently.
                                                                    # It re-calculates `tmp` but the result is not used before assigning map_locations.
                for i in tmp:
                    map_locations[i] = Lock() # This would re-create locks for existing locations, which is problematic.
                                            # Assuming `dev.map_locations = map_locations` above is the correct assignment.
                dev.map_locations = map_locations
                tmp = {}

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals timepoint completion.
        If a script is provided, it's appended to the device's internal script list
        and `script_received` event is set. If `script` is None, `timepoint_done`
        event is set.

        Args:
            script (object): The script object to be executed, or None to signal timepoint completion.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # If script is None, signal that script assignments for the timepoint are done.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's `sensor_data` dictionary.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
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
    This thread is responsible for fetching neighbor information, distributing scripts
    to dynamically created and managed `SingleDeviceThread`s in batches for concurrent
    execution, and participating in global synchronization.
    """

    def __init__(self, device):
        """
        Initializes a `DeviceThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
        """
        
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        It continuously fetches neighbor data, clears and waits for timepoint signals,
        spawns and manages `SingleDeviceThread`s for script execution,
        and participates in global synchronization.
        """
        
        while True:
            self.device.timepoint_done.clear() # Block Logic: Clear the timepoint_done event for the current cycle.
            neighbours = self.device.supervisor.get_neighbours() # Inline: Fetch neighbor information from the supervisor.
            if neighbours is None: # Inline: If `neighbours` is None, it signals termination for the device.
                break # Exit the main loop, terminating the DeviceThread.
            
            self.device.timepoint_done.wait() # Block Logic: Wait for the `timepoint_done` event to be set,
                                              # indicating that all scripts for the current timepoint have been assigned.
            script_list = [] # List to hold scripts for the current timepoint.
            thread_list = [] # List to hold spawned `SingleDeviceThread` instances.
            index = 0 # This index seems intended for distributing scripts, but its usage here is unusual.
                      # It only ever gets value 0 and is passed to `SingleDeviceThread` which then pops from `script_list`.
                      # The original code's logic here for `index` needs careful review.
            
            # Inline: Copy assigned scripts to a local list for processing.
            for script in self.device.scripts:
                script_list.append(script)
            
            # Block Logic: Spawns 8 `SingleDeviceThread`s.
            # This implementation assumes it always spawns 8 threads regardless of `len(script_list)`.
            # Each thread is passed the entire `script_list` and an `index`.
            # The `SingleDeviceThread`'s run method pops from `script_list` at `self.index`.
            # This logic is problematic for concurrent access to `script_list.pop(index)` without proper locking.
            for i in xrange(8):
                thread = SingleDeviceThread(self.device, script_list, neighbours, index)
                thread.start() # Start each worker thread.
                thread_list.append(thread) # Add to the list of managed threads.
            
            # Block Logic: Wait for all spawned `SingleDeviceThread`s to complete their execution.
            for i in xrange(len(thread_list)):
                thread_list[i].join()
            
            # Block Logic: Synchronize with other devices at the global barrier.
            # This ensures all devices have completed their script processing for the timepoint.
            self.device.barrier.wait()

class SingleDeviceThread(Thread):
    """
    A worker thread spawned by `DeviceThread` to execute a single script.
    It is designed to collect data from neighbors and its parent device,
    run the assigned script, and update sensor data, using a location-specific lock.

    NOTE: The current implementation of this thread's `run` method, specifically
    `self.script_list.pop(self.index)`, in conjunction with how it's initialized
    in `DeviceThread` (where `index` is always 0 for all spawned threads),
    suggests a potential race condition or unintended behavior. All `SingleDeviceThread`s
    might attempt to process the first script from the `script_list` simultaneously,
    leading to incorrect execution if `script_list` is modified by multiple threads without proper synchronization.
    """
    
    def __init__(self, device, script_list, neighbours, index):
        """
        Initializes a `SingleDeviceThread` instance.

        Args:
            device (Device): The parent `Device` object that spawned this thread.
            script_list (list): The list of scripts for the current timepoint. This list is shared
                                 among all `SingleDeviceThread`s spawned by `DeviceThread`.
            neighbours (list): A list of Device objects representing neighboring devices.
            index (int): The index of the specific script in `script_list` that this thread should execute.
                         (WARNING: In current `DeviceThread` implementation, this is always 0, leading to issues).
        """
        Thread.__init__(self)
        self.device = device
        self.script_list = script_list
        self.neighbours = neighbours
        self.index = index

    def run(self):
      
        if self.script_list != []:
            # WARNING: This `pop(self.index)` call is problematic if `self.index` is always 0
            # for multiple threads acting on the same `script_list` without external synchronization.
            # It implies all threads might try to process the same first script.
            (script, location) = self.script_list.pop(self.index)
            self.compute(script, location)

    def update(self, result, location):
        """
        Updates the sensor data of all neighboring devices (including the parent device)
        at a specific location with the given result.

        Args:
            result (any): The result to update the sensor data with.
            location (int): The data location to update.
        """
        # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.
        for device in self.neighbours:
            device.set_data(location, result) # Update neighbor's data.
        self.device.set_data(location, result) # Update this device's own data.

    def collect(self, location, neighbours, script_data):
        """
        Collects data from all neighboring devices and the parent device at a specific
        location, appending it to `script_data`. This operation is protected by
        acquiring the location-specific lock.

        Args:
            location (int): The data location from which to collect data.
            neighbours (list): A list of Device objects representing neighboring devices.
            script_data (list): The list to which collected data will be appended.
        """
        # Block Logic: Acquire the location-specific lock to ensure exclusive access
        # to data at this `location` during data collection.
        self.device.map_locations[location].acquire()
        # Block Logic: Collect data from all neighboring devices.
        for device in self.neighbours:
            data = device.get_data(location) # Get data from the neighbor.
            if data is None:
                pass # If no data, do nothing.
            else:
                script_data.append(data) # Add to script input if available.

        # Block Logic: Collect data from this worker's own parent device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data) # Add to script input if available.

    def compute(self, script, location):
        """
        Orchestrates the data collection, script execution, and data update for a single script.

        Args:
            script (object): The script object to be executed.
            location (int): The data location identifier to which the script applies.
        """
        script_data = [] # List to collect input data for the script.
        self.collect(location, self.neighbours, script_data) # Collect all necessary data.

        # Block Logic: If data was collected, execute the script and update device data.
        if script_data == []:
            pass # No data to compute with.
        else:
            # Inline: Execute the script's `run` method with the collected data.
            result = script.run(script_data)
            self.update(result, location) # Update device data with the result.

        self.device.map_locations[location].release() # Inline: Release the location-specific lock.

class ReusableBarrierSem():
    """
    A reusable barrier synchronization mechanism for multiple threads using semaphores.
    This barrier allows a fixed number of threads to wait at a synchronization point,
    and once all threads arrive, they are all released simultaneously. It can then
    be reused for subsequent synchronization points.
    """

    def __init__(self, num_threads):
        """
        Initializes the reusable barrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate
                               in the barrier synchronization.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock() # Lock to protect the shared counters.
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase of threads to wait on.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase of threads to wait on.

    def wait(self):
        """
        Causes the calling thread to wait until all threads have reached this barrier.
        This method orchestrates a two-phase synchronization to ensure reusability
        without deadlocks.
        """
        
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        The first phase of the barrier synchronization.
        Threads decrement a shared counter, and the last thread to reach zero
        releases all waiting threads for this phase.
        """
        
        with self.counter_lock: # Protect shared counter access.
            self.count_threads1 -= 1 # Decrement the count of threads remaining.
            if self.count_threads1 == 0: # If this is the last thread to arrive:
                for i in range(self.num_threads):
                    self.threads_sem1.release() # Release all waiting threads by incrementing the semaphore.

                self.count_threads1 = self.num_threads # Reset counter for next use.

        self.threads_sem1.acquire() # Wait (decrement) the semaphore, blocking until released by the last thread.

    def phase2(self):
        """
        The second phase of the barrier synchronization, necessary for reusability.
        Similar to phase1, threads decrement a counter, and the last thread
        releases all waiting threads for this phase.
        """
        
        with self.counter_lock: # Protect shared counter access.
            self.count_threads2 -= 1 # Decrement the count of threads remaining.
            if self.count_threads2 == 0: # If this is the last thread to arrive:
                for i in range(self.num_threads):
                    self.threads_sem2.release() # Release all waiting threads by incrementing the semaphore.
                self.count_threads2 = self.num_threads # Reset counter for next use.

        self.threads_sem2.acquire() # Wait (decrement) the semaphore, blocking until released by the last thread.