


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `ReusableBarrier`: A reusable barrier synchronization mechanism using semaphores stored in lists.
- `Device`: Represents a single device, managing its sensor data, communication with a supervisor,
  and orchestrating multi-threaded script execution.
- `NewThread`: A dedicated thread for executing a single script, collecting data,
  and updating device states.
- `DeviceThread`: The main orchestrating thread for a `Device`, responsible for fetching
  neighbor information, spawning `NewThread`s for each assigned script, and
  managing timepoint synchronization.

The system utilizes `threading.Event` for signaling, `threading.Lock` for protecting
shared resources (`location_lock` array), and custom barrier implementations for coordination.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier synchronization mechanism for multiple threads using semaphores.
    This barrier allows a fixed number of threads to wait at a synchronization point,
    and once all threads arrive, they are all released simultaneously. The counters
    are stored in lists (`count_threads1`, `count_threads2`) to allow for reusability
    of the barrier instance across multiple `wait` calls.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate
                               in the barrier synchronization.
        """
        self.num_threads = num_threads
        # Counters for the two phases of the barrier. Stored in lists to be mutable
        # when passed to the 'phase' method, enabling reusability.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock() # Lock to protect the shared counters during decrements and resets.
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase of threads to wait on.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase of threads to wait on.

    def wait(self):
        """
        Causes the calling thread to wait until all threads have reached this barrier.
        This method orchestrates a two-phase synchronization to ensure reusability
        without deadlocks.
        """
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current count of threads
                                  remaining for this phase.
            threads_sem (Semaphore): The semaphore threads wait on for this phase.
        """
        
        with self.count_lock: # Protect shared counter access.
            count_threads[0] -= 1 # Decrement the count of threads remaining.
            if count_threads[0] == 0: # If this is the last thread to arrive:
                for i in range(self.num_threads):
                    threads_sem.release() # Release all waiting threads by incrementing the semaphore.
                count_threads[0] = self.num_threads # Reset counter for next use.
        threads_sem.acquire() # Wait (decrement) the semaphore, blocking until released by the last thread.

class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution. It uses an array of locks
    for fine-grained control over data locations and manages worker threads.
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
        self.devices = [] # List to store references to all devices in the simulation.
        self.timepoint_done = Event() # Event to signal that the current timepoint's processing is complete.
        self.thread = DeviceThread(self) # The main orchestrating thread for this device.
        self.barrier = None # Placeholder for the global ReusableBarrier, set in setup_devices.
        self.list_thread = [] # List to hold spawned NewThread instances for script execution.
        self.thread.start() # Start the DeviceThread.
        self.location_lock = [None] * 100 # Array of locks for data locations, initialized with None.
                                          # Assuming a maximum of 100 locations.

    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes a global barrier and distributes it to all devices.
        It also populates this device's internal `devices` list with references
        to all other devices in the simulation. The barrier is created only once.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        
        # Block Logic: Initializes the global barrier only if it hasn't been set yet.
        # This typically means only one device (e.g., the master) will create it.
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices)) # Inline: Creates a global ReusableBarrier.
            self.barrier = barrier # Assigns the barrier to this device.
            # Inline: Distributes the created global barrier to all other devices that don't have it yet.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        # Block Logic: Populates this device's `devices` list with references to all other devices.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals timepoint completion.
        If a script is provided, it's appended to the device's internal script list.
        It also ensures that a `Lock` is associated with the given `location`
        in `self.location_lock`, either by finding an existing one from another
        device or creating a new one.

        Args:
            script (object): The script object to be executed, or None to signal timepoint completion.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        ok = 0 # Flag to indicate if a lock was found for the location from another device.
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
            # Block Logic: Initialize a Lock for the current location if one doesn't already exist.
            # This ensures that `location_lock[location]` points to a valid `Lock` object.
            if self.location_lock[location] is None:
                # Inline: Iterate through other devices to see if they already have a lock for this location.
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location] # Use the existing shared lock.
                        ok = 1 # Set flag to indicate a lock was found.
                        break
                if ok == 0: # If no existing lock was found for this location from other devices.
                    self.location_lock[location] = Lock() # Create a new Lock for this location.
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
        This implicitly triggers the shutdown of associated `NewThread`s managed by `DeviceThread`.
        """
        self.thread.join() # Wait for the DeviceThread to finish its execution.

class NewThread(Thread):
    """
    A worker thread responsible for executing a single script within a distributed device system.
    It collects necessary data from the parent device and its neighbors, runs the assigned script,
    and updates the sensor data on relevant devices, all while acquiring a location-specific lock.
    """
    
    def __init__(self, device, location, script, neighbours):
        """
        Initializes a `NewThread` instance.

        Args:
            device (Device): The parent `Device` object that spawned this thread.
            location (int): The data location identifier to which the script applies.
            script (object): The script object to be executed.
            neighbours (list): A list of Device objects representing neighboring devices.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        The main execution method for the `NewThread`.
        It acquires the location-specific lock, collects data from neighbors and
        its parent device, executes the assigned script, and then updates the
        sensor data on relevant devices with the script's result, before releasing the lock.
        """
        script_data = [] # List to collect input data for the script.
        # Block Logic: Acquire the location-specific lock to ensure exclusive access
        # to data at this `location` during script execution and data update.
        self.device.location_lock[self.location].acquire()
        
        # Block Logic: Collect data from all neighboring devices at the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location) # Get data from the neighbor.
            if data is not None:
                script_data.append(data) # Add to script input if available.
            
        # Block Logic: Collect data from this worker's own parent device at the specified location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data) # Add to script input if available.

        # Block Logic: If input data is available, execute the script and update device data.
        if script_data != []:
            # Inline: Execute the script's `run` method with the collected data.
            result = self.script.run(script_data)
            
            # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.
            for device in self.neighbours:
                device.set_data(self.location, result) # Update neighbor's data.
                
            self.device.set_data(self.location, result) # Update this device's own data.
        self.device.location_lock[self.location].release() # Release the location-specific lock.

class DeviceThread(Thread):
    """
    The main orchestrating thread for a `Device`.
    This thread is responsible for fetching neighbor information from the supervisor,
    waiting for timepoint completion signals, and then spawning individual `NewThread`s
    for each assigned script. It manages the execution and joining of these worker threads
    and participates in global synchronization.
    """

    def __init__(self, device):
        """
        Initializes a `DeviceThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        It continuously fetches neighbor data, waits for timepoint readiness,
        spawns `NewThread`s for each assigned script, waits for their completion,
        and participates in global synchronization.
        """
        while True:
            # Block Logic: Fetch neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Inline: If `neighbours` is None, it signals termination for the device.
            if neighbours is None:
                break # Exit the main loop, terminating the DeviceThread.

            # Block Logic: Wait for the `timepoint_done` event to be set,
            # indicating that all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Spawn `NewThread`s for each assigned script and manage their execution.
            # Each script is run in its own thread for concurrent processing.
            for (script, location) in self.device.scripts:
                thread = NewThread(self.device, location, script, neighbours) # Create a new `NewThread` instance.
                self.device.list_thread.append(thread) # Add to the list of managed script threads.

            for thread_elem in self.device.list_thread:
                thread_elem.start() # Start each `NewThread`.
            for thread_elem in self.device.list_thread:
                thread_elem.join() # Wait for each `NewThread` to complete its execution.
            self.device.list_thread = [] # Clear the list of threads for the next timepoint.

            # Block Logic: Clear the `timepoint_done` event for the next cycle.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronize with other devices at the global barrier.
            # This ensures all devices have completed their script processing for the timepoint.
            self.device.barrier.wait()