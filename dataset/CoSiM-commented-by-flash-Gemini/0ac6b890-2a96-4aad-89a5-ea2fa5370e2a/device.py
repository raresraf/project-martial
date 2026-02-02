"""
@file device.py
@brief This module defines classes for a distributed computational device simulation,
including `Device`, `DeviceThread`, and `RunScript`. It orchestrates concurrent
script execution, shared sensor data access, and synchronization across multiple
simulated devices.

Functional Utility: Provides a framework for simulating complex distributed systems,
allowing scripts to operate on local and remote sensor data while managing
concurrency and data consistency through various threading primitives.

Algorithm:
- `Device`: Represents a computational node that holds sensor data, manages a
  main `DeviceThread`, and assigns scripts. It uses semaphores and a `ReusableBarrier`
  for inter-device synchronization during setup and timepoint processing.
  It also maintains a dictionary of reentrant locks (`dislocksdict`) for
  location-specific data access across devices.
- `DeviceThread`: The primary worker thread for a `Device`. It continuously fetches
  neighbour information, waits for timepoint completion, and spawns `RunScript`
  threads to execute assigned scripts in parallel for each timepoint.
- `RunScript`: A short-lived thread responsible for executing a single assigned
  script for a specific location. It acquires location-specific locks before
  accessing data (both local and from neighbours) and propagates results back.
- Synchronization: Uses `ReusableBarrier` for global timepoint synchronization,
  `threading.Event` to signal script readiness, `threading.Semaphore` for setup
  coordination, and `threading.Lock` (reentrant and standard) for protecting
  shared data during script execution and data access.

Time Complexity:
- `Device.setup_devices`: O(D*L) for lock initialization, where D is number of devices, L is number of locations.
- `DeviceThread.run`: The main loop runs indefinitely. Within each timepoint, it involves
  `ReusableBarrier.wait()` (O(1) average), `Event.wait()`, and then spawning and joining
  `S` number of `RunScript` threads.
- `RunScript.run`: Involves acquiring/releasing a reentrant lock (O(1)) and iterating
  through `N` neighbours to get/set data (O(N)), each protected by a standard lock (O(1)).
  So, O(N) per script execution.
- Total time per timepoint is roughly O(S * N) where S is number of scripts and N is neighbours.
Space Complexity:
- `Device`: O(L) for sensor data, O(S) for scripts, O(L_total) for shared `dislocksdict`, O(1) for other sync primitives.
- `RunScript`: O(1) beyond references.
- `DeviceThread`: O(S) for managing `script_threads`.
"""


from threading import Event, Thread, Semaphore, Lock, RLock
from reusable_barrier import ReusableBarrier
import multiprocessing # Imported but not explicitly used in the provided code snippet.

class Device(object):
    """
    @class Device
    @brief Represents a single computational node in a distributed system simulation.
    Functional Utility: Manages its local sensor data, orchestrates script execution
    through a main `DeviceThread`, and coordinates with a supervisor and other devices
    using various synchronization primitives. It also handles data consistency via locks.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id (int): A unique identifier for this device.
        @param sensor_data (dict): A dictionary representing sensor data local to this device.
                                  Keys are location IDs, values are data.
        @param supervisor (object): A reference to the supervisor object responsible for
                                    global coordination (e.g., getting neighbours).
        """
        
        self.results = {} # Dictionary to store results of script execution.
        self.lock = None # Standard Lock, used for protecting local data access during `get_data`/`set_data` by `RunScript` threads.
        self.dislocksdict = None # Dictionary of Reentrant Locks (RLocks) for location-specific data access across devices.
        self.barrier = None # Global reusable barrier for synchronizing all devices.
        self.sem = Semaphore(1) # Unused semaphore in this specific code context.
        self.sem2 = Semaphore(0) # Semaphore used for coordinating setup among devices.
        self.all_devices = [] # List of all devices in the simulation.
        self.device_id = device_id # Unique identifier for this device.
        self.sensor_data = sensor_data # Dictionary storing local sensor data.
        self.supervisor = supervisor # Reference to the supervisor object.
        self.script_received = Event() # Event to signal when scripts have been received (deprecated by timepoint_done).
        self.scripts = [] # List of (script, location) tuples assigned to this device for execution.
        self.timepoint_done = Event() # Event to signal when all scripts for a timepoint have been assigned.
        self.thread = DeviceThread(self) # The main worker thread for this device.
        self.thread.start() # Start the main `DeviceThread` worker.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return (str): A formatted string indicating the device ID.
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives across all devices in the system.
        Functional Utility: This method ensures that all devices share the same global
        `ReusableBarrier`, the same dictionary of location-specific `RLock`s (`dislocksdict`),
        and a standard `Lock` (`lock`) for individual device data. The setup is coordinated
        to be performed only once by the first device (`device_id == 0`) and then propagated.
        @param devices (list): A list of all `Device` instances in the simulation.
        """
        
        loc = [] # List to collect all unique locations across all devices.
        for d in devices: # Block Logic: Collect all unique locations.
            for l in d.sensor_data:
                loc.append(l) 
        self.all_devices = devices # Store the list of all devices.

        if self.device_id == 0: # Block Logic: Only the first device (ID 0) performs the initial setup.
            self.sem2.release() # Release semaphore to signal other devices can proceed.
            self.barrier = ReusableBarrier(len(devices)) # Initialize the global barrier with the total number of devices.
            self.dislocksdict = {} # Initialize the dictionary for distributed location locks.
            for k in list(set(loc)): # Block Logic: Create an RLock for each unique location.
                self.dislocksdict[k] = RLock()
            self.lock = Lock() # Initialize a standard lock for local device data.

        self.sem2.acquire() # Synchronization Point: Wait for the first device to complete its setup.

        # Block Logic: Propagate the shared resources (barrier, dislocksdict, lock) to other devices.
        for d in devices:
            if d.barrier == None: # If the barrier hasn't been set yet for a device.
                d.barrier = self.barrier # Assign the global barrier.
                d.sem2.release() # Release semaphore to unblock this device's setup.
                d.dislocksdict = self.dislocksdict # Assign the shared dictionary of location locks.
                d.lock = Lock() # Assign a new standard lock for local device data.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location, or signals timepoint completion.
        Functional Utility: Adds a script-location pair to the device's execution queue.
        If `script` is `None`, it signals that all scripts for the current
        timepoint have been assigned and the device's thread can proceed.
        @param script (object): The script object to assign. If `None`, signals end of timepoint assignment.
        @param location (object): The location ID where the script should be executed.
        """
        
        if script is not None: # Block Logic: If a script is provided.
            self.scripts.append((script, location)) # Add script to the list.
        else: # Block Logic: If `script` is `None`, signal timepoint completion.
            self.timepoint_done.set() # Set the event to unblock the `DeviceThread`.
   
    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Functional Utility: Provides the value of sensor data for a specified location
        if it exists within the device's local data.
        @param location (object): The location ID of the sensor data to retrieve.
        @return (object or None): The sensor data at the specified location, or `None` if not found.
        """
        
        data = -1 # Initialize data to a default invalid value.
        if location in self.sensor_data: # Block Logic: Check if the location exists in local sensor data.
            data = self.sensor_data[location] # Retrieve data.
            return data
        else:
            return None # Return None if location not found.

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
        Functional Utility: Updates the value of sensor data for a specified location
        if it exists within the device's local data.
        @param location (object): The location ID where the sensor data should be set.
        @param data (object): The new sensor data value.
        """
        
        
        if location in self.sensor_data: # Block Logic: Check if the location exists in local sensor data.
            self.sensor_data[location] = data # Update data if location is local.

    def shutdown(self):
        """
        @brief Shuts down the main `DeviceThread` associated with this device.
        Functional Utility: Waits for the `DeviceThread` to complete its execution,
        ensuring a clean termination of the device's main operations. This typically
        implies that the thread has received a signal to break its main loop.
        """
        
        self.thread.join() # Wait for the main thread to finish.

class RunScript(Thread):
    """
    @class RunScript
    @brief A short-lived worker thread responsible for executing a single script.
    Functional Utility: This thread is spawned by `DeviceThread` to handle the execution
    of one specific script. It acquires location-specific reentrant locks and
    standard device locks to ensure thread-safe access to sensor data, collects data
    from the current and neighbouring devices, runs the script, and propagates the results.
    """
    def __init__(self, script, location, neighbours, device):
        """
        @brief Initializes a new `RunScript` instance.
        @param script (object): The script object to execute.
        @param location (object): The location ID for which this script is to be run.
        @param neighbours (list): A list of neighbouring devices to fetch data from.
        @param device (Device): The parent `Device` instance.
        """
        Thread.__init__(self) # Call the base Thread constructor.
        self.script = script # Script to execute.
        self.location = location # Location ID for the script.
        self.neighbours = neighbours # List of neighbouring devices.
        self.device = device # Reference to the parent device.
    def run(self):
        """
        @brief The main execution method for `RunScript`.
        Functional Utility: Acquires a reentrant lock for the specific location, then
        acquires standard locks for each device (current and neighbours) to protect
        `get_data` and `set_data` calls. It collects data, executes the assigned script,
        updates the data on the current and neighbouring devices, and finally
        releases all acquired locks.
        """
        
        # Precondition: `self.device.dislocksdict[self.location]` exists and is an RLock.
        # Functional Utility: Ensures exclusive write access to the shared location data across devices.
        self.device.dislocksdict[self.location].acquire()
        script_data = [] # List to accumulate data for the script.
        
        # Block Logic: Collect data from neighbouring devices.
        for device in self.neighbours:  
            device.lock.acquire() # Functional Utility: Acquire lock to protect `get_data` on the neighbour device.
            data = device.get_data(self.location) # Retrieve data from neighbour.
            device.lock.release() # Release lock on neighbour device.
            if data is not None:
                script_data.append(data)
                
        self.device.lock.acquire() # Functional Utility: Acquire lock to protect `get_data` on the current device.
        data = self.device.get_data(self.location) # Retrieve local data.
        self.device.lock.release() # Release lock on current device.
        if data is not None:
            script_data.append(data)


        if script_data != []: # Block Logic: If any data was collected, run the script and update results.
            # Functional Utility: Execute the script's `run` method with the collected data.
            result = self.script.run(script_data) 
            
            # Block Logic: Propagate the script's result to neighbouring devices.
            for device in self.neighbours:
                device.lock.acquire() # Acquire lock to protect `set_data` on the neighbour device.
                device.set_data(self.location, result) # Update neighbour's data.
                device.lock.release() # Release lock on neighbour device.
            self.device.lock.acquire() # Acquire lock to protect `set_data` on the current device.
            self.device.set_data(self.location, result) # Update local sensor data with the result.
            self.device.lock.release() # Release lock on current device.
        # Postcondition: The RLock for `self.location` is released.
        self.device.dislocksdict[self.location].release()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief The main worker thread for a `Device`.
    Functional Utility: This thread runs continuously, coordinating the activities
    of its parent `Device`. It fetches global state (neighbours), waits for script
    assignments (`timepoint_done`), and then spawns multiple `RunScript` instances
    to concurrently process these scripts for a given timepoint, synchronizing
    at global barriers.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new `DeviceThread` instance.
        @param device (Device): The parent `Device` instance to which this thread belongs.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Set a descriptive thread name.
        self.device = device # Reference to the parent device.


    def run(self):
        """
        @brief The main execution loop for the `DeviceThread`.
        Functional Utility: Manages the device's timepoint processing. It continuously
        fetches neighbour information, synchronizes with other device threads using a
        global barrier, waits for scripts to be ready, and then executes these scripts
        by spawning and joining `RunScript` threads.
        """
        while True: # Block Logic: Main loop for continuous timepoint processing until shutdown.
            
            # Functional Utility: Fetch updated neighbour information from the supervisor.
            # Precondition: `self.device.supervisor` is a valid object with a `get_neighbours` method.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Block Logic: Check for shutdown signal (neighbours being None).
                break # Exit the loop and terminate the thread if shutdown is signaled.
            self.device.timepoint_done.wait() # Synchronization Point: Wait for signal that scripts for timepoint are assigned.


            self.device.barrier.wait() # Synchronization Point: Global barrier for all devices to start processing a timepoint.
            script_threads = [] # List to hold `RunScript` threads for current timepoint.
            for (script, location) in self.device.scripts: # Block Logic: Create a `RunScript` thread for each assigned script.
                script_threads.append(RunScript(script, location, neighbours, self.device))
            for t in script_threads: # Block Logic: Start all script execution threads.
                t.start() 
            for t in script_threads: # Block Logic: Wait for all script execution threads to complete.
                t.join() 
            self.device.barrier.wait() # Synchronization Point: Global barrier for all devices to finish processing a timepoint.
            self.device.timepoint_done.clear() # Clear the timepoint_done event for the next timepoint.