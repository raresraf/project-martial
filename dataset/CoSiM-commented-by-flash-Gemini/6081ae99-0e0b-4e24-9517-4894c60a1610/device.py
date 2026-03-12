


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `Barrier`: A reusable barrier synchronization mechanism using `threading.Condition`.
- `Device`: Represents a single device, managing its sensor data and orchestrating operations.
- `DeviceThread`: Worker threads for a `Device`, with a specialized "first" thread
  handling coordinating duties like fetching neighbors and resetting events.

The system utilizes global static members (`Device.bariera_devices` and `Device.locks`)
for inter-device and data-location synchronization, along with per-device events (`Event`)
and locks (`Lock`) for internal management.
"""

from threading import Event, Thread, Condition, Lock


class Barrier(object):
    """
    A reusable barrier synchronization mechanism for multiple threads using `threading.Condition`.
    Threads wait at this barrier until a specified number of threads (`num_threads`) have arrived.
    Once all threads arrive, they are all notified and released simultaneously.
    The barrier can then be reused for subsequent synchronization points.
    """
    def __init__(self, num_threads=0):
        """
        Initializes the reusable barrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate
                               in the barrier synchronization. Defaults to 0,
                               implying it might be set later or is a placeholder.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Counter for threads currently waiting at the barrier.
        
        self.cond = Condition() # The condition variable used for thread synchronization.

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have arrived. Once all threads are present, they are all released.
        """
        
        # Block Logic: Acquire the condition variable's intrinsic lock.
        self.cond.acquire()
        self.count_threads -= 1 # Decrement the count of threads waiting.
        if self.count_threads == 0: # If this is the last thread to arrive:
            
            self.cond.notify_all() # Notify all waiting threads to resume.
            self.count_threads = self.num_threads # Reset the counter for the next use of the barrier.
        else:
            # If not the last thread, wait to be notified.
            self.cond.wait()
        
        self.cond.release() # Release the condition variable's intrinsic lock.

class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution. It utilizes global
    static barriers and locks for inter-device synchronization and data consistency.
    """
    
    # Static member: A global barrier for synchronizing all `Device` instances.
    bariera_devices = Barrier()
    # Static member: A global list of locks, one for each data location, shared across all devices.
    locks = []

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

        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).
        self.locations = [] # List to hold locations associated with assigned scripts.
        
        self.nr_scripturi = 0 # Counter for the number of assigned scripts.
        self.script_crt = 0 # Index of the current script being processed (shared among DeviceThreads).

        self.timepoint_done = Event() # Event to signal that the current timepoint's processing is complete.

        
        self.neighbours = [] # List to store references to neighboring devices.
        self.event_neighbours = Event() # Event to signal that neighbor information has been fetched.
        self.lock_script = Lock() # Lock to protect `script_crt` and `nr_scripturi` access.
        self.bar_thr = Barrier(8) # A barrier for synchronizing the 8 `DeviceThread` instances within this Device.

        # Block Logic: Create and start the "first" DeviceThread (coordinator).
        self.thread = DeviceThread(self, 1) # 'first' parameter set to 1 for coordinator.
        self.thread.start()
        self.threads = [] # List to hold the other 7 worker DeviceThread instances.
        # Block Logic: Create and start the other 7 worker DeviceThread instances.
        for _ in range(7):
            tthread = DeviceThread(self, 0) # 'first' parameter set to 0 for worker.
            self.threads.append(tthread)
            tthread.start()

    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the static `Device.bariera_devices` and `Device.locks` for global use.
        This method is designed to be called only by the device with `device_id == 0`.
        It creates a global barrier for all devices and initializes a list of locks
        for each possible data location.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        # Block Logic: The global device barrier is initialized with the total number of devices.
        Device.bariera_devices = Barrier(len(devices))
        
        # Block Logic: Initializes a global list of locks, one for each data location.
        # This block ensures that `Device.locks` is populated only once.
        if Device.locks == []:
            # Inline: The number of locations is obtained from the supervisor's testcase configuration.
            for _ in range(self.supervisor.supervisor.testcase.num_locations):
                Device.locks.append(Lock()) # Appends a new `Lock` for each location.

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals timepoint completion.
        If a script is provided, it's appended to the device's internal lists.
        If `script` is None, it signals that script assignments for the current
        timepoint are complete.

        Args:
            script (object): The script object to be executed, or None to signal timepoint completion.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append(script) # Add the script object to the list.
            self.locations.append(location) # Add the location to the list.
            
            self.nr_scripturi += 1 # Increment the count of assigned scripts.
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
        return self.sensor_data[location] if location in \
        self.sensor_data else None

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
        Initiates the shutdown process for the device by waiting for its main `DeviceThread`
        and all its worker `DeviceThread` instances to complete their execution.
        """
        self.thread.join() # Wait for the primary DeviceThread (coordinator) to finish.
        for tthread in self.threads:
            tthread.join() # Wait for all worker DeviceThreads to finish.


class DeviceThread(Thread):
    """
    A worker thread associated with a `Device`.
    One `DeviceThread` (`first=1`) acts as a coordinator, fetching neighbor information
    and resetting events. Other `DeviceThread`s (`first=0`) act as workers,
    processing a subset of scripts by acquiring and releasing locks for data consistency,
    executing scripts, and updating sensor data.
    """
    
    def __init__(self, device, first):
        """
        Initializes a `DeviceThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
            first (int): A flag (1 for coordinator, 0 for worker) indicating the thread's role.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.first = first

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        This method's behavior differs based on the `first` flag.
        The "first" thread (coordinator) fetches neighbor information and resets events.
        All threads process scripts, acquiring location-specific locks,
        collecting data, executing scripts, and updating sensor data.
        Synchronization occurs at a local thread barrier and a global device barrier.
        """
        while True:
            # Block Logic: Coordinating duties performed only by the "first" thread.
            # This thread fetches neighbors and resets the shared script counter.
            if self.first == 1:
                # Inline: Fetch neighbor information from the supervisor.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                # Inline: Reset the shared script counter for the new timepoint.
                self.device.script_crt = 0
                # Inline: Signal that neighbor information has been fetched for this timepoint.
                self.device.event_neighbours.set()

            # Block Logic: All threads wait until neighbor information is available.
            self.device.event_neighbours.wait()

            # Inline: If `neighbours` is None, it signals termination for the device.
            if self.device.neighbours is None:
                break # Exit the main loop, terminating the DeviceThread.

            # Block Logic: Wait for the `timepoint_done` event to be set,
            # indicating that all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Loop to process scripts assigned to this device.
            # Scripts are processed sequentially, but multiple DeviceThreads might be processing
            # different scripts concurrently.
            while True:
                # Block Logic: Acquire `lock_script` to safely get and increment `script_crt`.
                # This ensures each script is processed only once across all `DeviceThread`s.
                self.device.lock_script.acquire()
                index = self.device.script_crt # Get the index of the next script to process.
                self.device.script_crt += 1 # Increment the shared counter.
                self.device.lock_script.release() # Release the lock.

                # Inline: If the current index exceeds the number of assigned scripts, all scripts are processed.
                if index >= self.device.nr_scripturi:
                    break # Exit the script processing loop.

                # Inline: Retrieve the script and its associated location using the current index.
                location = self.device.locations[index]
                script = self.device.scripts[index]

                # Block Logic: Acquire the global lock for the specific location to ensure exclusive access
                # during data collection and update operations.
                Device.locks[location].acquire()

                script_data = [] # List to collect input data for the script.
                    
                # Block Logic: Collect data from all neighboring devices at the specified location.
                for device in self.device.neighbours:
                    data = device.get_data(location) # Get data from the neighbor.
                    if data is not None:
                        script_data.append(data) # Add to script input if available.
                
                # Block Logic: Collect data from this worker's own parent device at the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data) # Add to script input if available.

                # Block Logic: If input data is available, execute the script and update device data.
                if script_data != []:
                        
                    result = script.run(script_data) # Execute the script with the collected data.

                    # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.
                    for device in self.device.neighbours:
                        device.set_data(location, result) # Update neighbor's data.
                    
                    self.device.set_data(location, result) # Update this device's own data.

                Device.locks[location].release() # Release the global lock for this location.

            # Block Logic: First synchronization point for all `DeviceThread`s within this Device.
            # Ensures all local threads have finished processing their share of scripts for the timepoint.
            self.device.bar_thr.wait()
            
            # Block Logic: Coordinating duties performed only by the "first" thread.
            # This thread clears events for the next timepoint.
            if self.first == 1:
                self.device.event_neighbours.clear() # Clear neighbor fetched event.
                self.device.timepoint_done.clear() # Clear timepoint done event.
            # Block Logic: Second synchronization point for all `DeviceThread`s within this Device.
            # Ensures events are cleared before proceeding.
            self.device.bar_thr.wait()
            
            # Block Logic: Global synchronization point for all devices.
            # Only the "first" thread of each device waits on the global barrier.
            if self.first == 1:
                Device.bariera_devices.wait() # Wait on the global device barrier for all devices to finish.

