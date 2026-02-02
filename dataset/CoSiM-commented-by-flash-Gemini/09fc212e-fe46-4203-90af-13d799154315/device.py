"""
@file device.py
@brief This module simulates a distributed computational environment using a multi-threaded
approach. It defines `Device` objects, each managing its own `DeviceThread` (main worker)
and spawning `MyThread` instances to execute scripts. Synchronization is handled
through custom barrier and semaphore implementations, along with standard threading locks.

Algorithm:
- `ReusableBarrierSem`: A custom barrier implementation using semaphores to synchronize
  multiple threads across two phases, allowing for reuse.
- `Device`: Represents a computational node. It holds sensor data, manages a main `DeviceThread`,
  and dynamically creates `MyThread` instances to run scripts. It centralizes synchronization
  primitives (`locks`, `barrier`) and coordinates with a `supervisor` for global state.
- `MyThread`: A short-lived worker thread responsible for executing a single script at a given location.
  It acquires and releases locks for data access.
- `DeviceThread`: The main worker thread for a `Device`. It continuously fetches neighbour
  information, waits for scripts to be assigned, spawns `MyThread` instances for each script,
  and synchronizes at global barriers.

Time Complexity:
- `ReusableBarrierSem.wait`: O(num_threads) due to semaphore releases in each phase.
- `Device.setup_devices`: O(D) to configure barriers and O(L_total) to collect all location locks, where D is number of devices, L_total is total locations.
- `Device.assign_script`: O(L_total) in worst case to find existing lock for a location.
- `DeviceThread.run`: The main loop runs indefinitely. Within each timepoint, it iterates over scripts (S),
  spawns `MyThread`s which then acquire/release locks, leading to operations proportional to S * (L_acquire + L_release + N * L_access), where N is number of neighbours.
Space Complexity:
- `Device`: O(L) for sensor data, O(S) for scripts, O(L_total) for location locks, O(N) for neighbours.
- `ReusableBarrierSem`: O(1).
- `MyThread`: O(1) beyond references.
- `DeviceThread`: O(S) for temporary list of `MyThread`s.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrierSem():
    """
    @class ReusableBarrierSem
    @brief A custom reusable barrier implementation using semaphores.
    Functional Utility: Synchronizes a fixed number of threads in two phases, ensuring
    all threads reach specific points before any can proceed. This barrier can be reset
    and used multiple times.
    """
    
    
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrierSem.
        @param num_threads (int): The number of threads that must call `wait()` before
                                  any can proceed.
        """
        self.num_threads = num_threads # Total number of threads to synchronize.
        self.count_threads1 = self.num_threads # Counter for the first phase.
        self.count_threads2 = self.num_threads # Counter for the second phase.
        self.counter_lock = Lock() # Lock to protect access to the counters.
        self.threads_sem1 = Semaphore(0) # Semaphore for releasing threads in phase 1.
        self.threads_sem2 = Semaphore(0) # Semaphore for releasing threads in phase 2.
    
    def wait(self):
        """
        @brief Waits at the barrier.
        Functional Utility: Causes the calling thread to block until all `num_threads`
        have called `wait()`, completing both synchronization phases.
        """
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """
        @brief First synchronization phase.
        Functional Utility: Threads decrement a counter. The last thread to reach zero
        releases all other waiting threads for phase 1.
        """
        with self.counter_lock: # Acquire lock to safely decrement counter.
            self.count_threads1 -= 1 # Decrement thread counter for phase 1.
            if self.count_threads1 == 0: # If this is the last thread.
                for i in range(self.num_threads): # Release all threads waiting on semaphore 1.
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset counter for next use.
        self.threads_sem1.acquire() # Acquire semaphore 1, blocking until released by last thread.
    
    def phase2(self):
        """
        @brief Second synchronization phase.
        Functional Utility: Threads decrement a counter. The last thread to reach zero
        releases all other waiting threads for phase 2.
        """
        with self.counter_lock: # Acquire lock to safely decrement counter.
            self.count_threads2 -= 1 # Decrement thread counter for phase 2.
            if self.count_threads2 == 0: # If this is the last thread.
                for i in range(self.num_threads): # Release all threads waiting on semaphore 2.
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for next use.
        self.threads_sem2.acquire() # Acquire semaphore 2, blocking until released by last thread.


class Device(object):
    """
    @class Device
    @brief Represents a single computational node in a distributed simulation.
    Functional Utility: Manages local sensor data, orchestrates script execution via
    `MyThread` workers, and handles complex synchronization across timepoints and
    with other devices. It holds its own `DeviceThread` (main worker) and manages
    shared resources like global barriers and per-location locks.
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
        self.device_id = device_id # Unique identifier for this device.
        self.sensor_data = sensor_data # Dictionary storing local sensor data.
        self.supervisor = supervisor # Reference to the supervisor object.
        self.script_received = Event() # Event to signal when a script has been received.
        self.scripts = [] # List of (script, location) tuples assigned to this device for execution.
        self.timepoint_done = Event() # Event to signal when all scripts for a timepoint have been assigned.
        self.thread = DeviceThread(self) # The main worker thread for this device.

        
        self.neighbours = [] # List of neighbouring devices, updated by the supervisor.
        self.alldevices = [] # List of all devices in the simulation.
        self.barrier = None # Global barrier for synchronizing all devices.
        self.threads = [] # List of `MyThread` worker threads spawned for a timepoint.
        self.threads_number = 8 # Number of worker threads for this device.
        self.locks = [None] * 100 # Array to hold RLock objects for data locations.

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
        `ReusableBarrierSem` and the same set of per-location locks, centralizing
        coordination for the entire simulation. It configures these resources once
        and then propagates them.
        @param devices (list): A list of all `Device` instances in the simulation.
        """
        
        # Block Logic: Initialize the global barrier if it hasn't been set yet.
        # This block ensures that only one `ReusableBarrierSem` instance is created
        # and shared among all devices, using the first device encountered to manage it.
        if self.barrier is None:
            # Precondition: `self.barrier` is `None`.
            # Postcondition: `self.barrier` is initialized and shared across all devices.
            barrier = ReusableBarrierSem(len(devices)) # Create a new reusable barrier for all devices.
            self.barrier = barrier # Assign the newly created barrier to this device.
            for d in devices: # Propagate the barrier to all other devices.
                if d.barrier is None:
                    d.barrier = barrier
        
        # Block Logic: Collect all devices into a local list.
        # Functional Utility: Provides the `DeviceThread` with a complete list of all
        # devices in the simulation, which might be needed for certain operations.
        for device in devices:
            if device is not None:
                self.alldevices.append(device)


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location, or signals timepoint completion.
        Functional Utility: Adds a script-location pair to the device's execution queue.
        It also handles the initialization of `RLock`s for data locations and signals
        the `DeviceThread` when scripts are available or when a timepoint is done.
        @param script (object): The script object to assign. If `None`, it signals completion of a timepoint.
        @param location (object): The location ID where the script should be executed.
        """
        
        
        no_lock_for_location = 0; # Flag to track if a lock was found for the location.
        if script is not None: # Case 1: A script is being assigned.
            self.scripts.append((script, location)) # Add the script to the device's list.
            # Block Logic: Check if a lock for this location already exists in another device.
            # Functional Utility: Ensures that only one RLock object exists for each location ID
            # across all devices, preventing redundant lock creation and potential deadlocks.
            for device in self.alldevices:
                if device.locks[location] is not None:
                    self.locks[location] = device.locks[location] # Use existing lock.
                    no_lock_for_location = 1; # Set flag.
                    break;
            # Block Logic: If no existing lock was found, create a new one for this location.
            if no_lock_for_location == 0:
                self.locks[location] = Lock() # Create a new lock.
            self.script_received.set() # Signal that a script has been received.
        else: # Case 2: Signal timepoint completion.
            self.timepoint_done.set() # Set the 'timepoint_done' event.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location (object): The location ID of the sensor data to retrieve.
        @return (object or None): The sensor data at the specified location, or `None` if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None # Return data if local, else None.

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
        @param location (object): The location ID where the sensor data should be set.
        @param data (object): The new sensor data value.
        """
        if location in self.sensor_data: # Block Logic: Check if the location exists in local sensor data.
            self.sensor_data[location] = data # Update data if location is local.

    def shutdown(self):
        """
        @brief Shuts down the main `DeviceThread` associated with this device.
        Functional Utility: Waits for the `DeviceThread` to complete its execution,
        ensuring a clean termination of the device's main operations.
        """
        self.thread.join() # Wait for the main thread to finish.


class MyThread(Thread):
    """
    @class MyThread
    @brief A short-lived worker thread responsible for executing a single script.
    Functional Utility: This thread is spawned by `DeviceThread` to handle the execution
    of one specific script. It acquires and releases location-specific locks to ensure
    thread-safe access to sensor data, collects data from neighbours, runs the script,
    and propagates the results.
    """

    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes a new `MyThread` instance.
        @param device (Device): The parent `Device` instance.
        @param location (object): The location ID for which this script is to be run.
        @param script (object): The script object to execute.
        @param neighbours (list): A list of neighbouring devices to fetch data from.
        """
        Thread.__init__(self) # Call the base Thread constructor.
        self.device = device # Reference to the parent device.
        self.location = location # Location ID for the script.
        self.script = script # Script to execute.
        self.neighbours = neighbours # List of neighbouring devices.

    def run(self):
        """
        @brief The main execution method for `MyThread`.
        Functional Utility: Acquires a lock for the specific location, collects data
        from the current and neighbouring devices, executes the assigned script,
        updates the data on the current and neighbouring devices, and finally
        releases the lock.
        """
        # Precondition: `self.device.locks[self.location]` must be initialized.
        # Functional Utility: Ensures exclusive access to the data at `self.location`.
        self.device.locks[self.location].acquire() # Acquire lock for the location.
        script_data = [] # List to accumulate data for the script.
        
        # Block Logic: Collect data from neighbouring devices.
        for device in self.neighbours:
            data = device.get_data(self.location) # Retrieve data from neighbour.
            if data is not None:
                script_data.append(data) # Add data if available.
        
        # Block Logic: Collect data from the current device.
        data = self.device.get_data(self.location) # Retrieve local data.
        if data is not None:
            script_data.append(data) # Add local data if available.

        if script_data != []: # Block Logic: If data is available, run the script.
            
            # Functional Utility: Executes the script's `run` method with the collected data.
            result = self.script.run(script_data)

            
            # Functional Utility: Propagates the script's result to neighbouring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result) # Update local sensor data with the result.
        # Postcondition: The lock for `self.location` is released, making the data accessible to others.
        self.device.locks[self.location].release() # Release lock for the location.

class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief The main worker thread for a `Device`.
    Functional Utility: This thread runs continuously, coordinating the activities
    of its parent `Device`. It fetches global state (neighbours), waits for script
    assignments, and then spawns multiple `MyThread` instances to concurrently
    process these scripts for a given timepoint, synchronizing at global barriers.
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
        Functional Utility: This loop manages the device's timepoint processing.
        It continuously fetches neighbour information, waits for scripts, and then
        orchestrates the execution of those scripts by spawning `MyThread` instances.
        It also handles global synchronization at the end of each timepoint.
        """
        while True: # Block Logic: Main loop for continuous timepoint processing until shutdown.
            
            # Functional Utility: Fetch updated neighbour information from the supervisor.
            # Precondition: `self.device.supervisor` is a valid object with a `get_neighbours` method.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Block Logic: Check for shutdown signal (neighbours being None).
                break # Exit the loop and terminate the thread if shutdown is signaled.

            self.device.timepoint_done.wait() # Synchronization Point: Wait for signal that scripts for timepoint are assigned.

            
            self.device.neighbours = neighbours # Update the device's list of neighbours.

            count = 0 # Counter for assigned scripts.
            
            # Block Logic: Iterate through assigned scripts and spawn `MyThread`s for execution.
            # Functional Utility: Distributes the execution of each script to a dedicated `MyThread`.
            # Invariant: Each script is assigned to a `MyThread` up to `self.device.threads_number` (8).
            for (script, location) in self.device.scripts:
                
                if count >= self.device.threads_number: # Limit the number of concurrently running `MyThread`s.
                    break
                count = count + 1 # Increment count of spawned threads.
                thread = MyThread(self.device, location, script, neighbours) # Create a new `MyThread`.
                self.device.threads.append(thread) # Add to list of current worker threads.

            
            # Block Logic: Start and join all spawned `MyThread`s.
            # Functional Utility: Ensures all scripts for the current batch are processed
            # before moving to the next timepoint.
            for thread in self.device.threads:
                thread.start() # Start the `MyThread`.
            for thread in self.device.threads:
                thread.join() # Wait for the `MyThread` to complete.
            self.device.threads = [] # Clear the list of spawned threads.

            
            # Postcondition: The current timepoint is processed, and the device is ready for the next.
            self.device.timepoint_done.clear() # Clear the 'timepoint_done' event for the next timepoint.
            # Synchronization Point: Global barrier. All devices wait here after processing their timepoint.
            # Functional Utility: Ensures that no device proceeds to the next timepoint before all
            # other devices have finished their current timepoint processing.
            self.device.barrier.wait()