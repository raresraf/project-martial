"""
@file device.py
@brief This module simulates a distributed computational environment using a multi-threaded
approach. It defines `Device` objects, each managing its own `DeviceThread` (main worker)
and uses a custom `Barrier` for synchronization. This setup allows for processing
scripts over shared sensor data, simulating a system where devices cooperate.

Algorithm:
- `Barrier`: A custom reusable barrier implementation using `threading.Condition` and
  `threading.Event` to synchronize multiple threads. It ensures all participating
  threads reach a synchronization point before any can proceed.
- `Device`: Represents a computational node. It manages local sensor data, holds a
  main `DeviceThread`, and assigns scripts for execution. All devices share a single
  global `Barrier` instance for inter-device synchronization at timepoints.
- `DeviceThread`: The main worker thread for a `Device`. It continuously fetches
  neighbour information, waits for scripts to be assigned, and then executes these
  scripts. Data access to sensor data (local and remote/neighbour) is performed
  without explicit per-location locks in this version, implying a specific data
  consistency model or external guarantees.

Time Complexity:
- `Barrier.wait`: O(1) in the average case (thread acquisition/release).
- `Device.assign_script`: O(1).
- `DeviceThread.run`: The main loop runs indefinitely, processing timepoints.
  Within each timepoint, script processing depends on the number of scripts (S)
  and neighbours (N). Script execution involves iterating over neighbours to collect
  data and then propagating results, leading to O(S * N) operations per timepoint.
Space Complexity:
- `Device`: O(L) for sensor data, O(S) for scripts, O(1) for shared barrier.
- `Barrier`: O(1) for internal state.
- `DeviceThread`: O(1) beyond references to the parent device and shared resources.
"""

from threading import Event, Thread, Condition


class Barrier():
    """
    @class Barrier
    @brief A simple reusable barrier implementation for synchronizing multiple threads.
    Functional Utility: Ensures that a specified number of threads (`num_threads`)
    all reach a certain point in their execution before any of them are allowed
    to proceed. It uses a `threading.Condition` for waiting/notifying and
    resets its internal counter for reuse.
    """
    
    # Class variables to store the total number of threads and the current count of waiting threads.
    # These are static because all Barrier instances synchronize the same set of threads.
    num_threads = 0
    count_threads = 0

    def __init__(self):
        """
        @brief Initializes the Barrier.
        Functional Utility: Sets up the internal `Condition` object used for thread synchronization.
        """
        
        self.cond = Condition() # Condition variable for blocking and unblocking threads.
        self.thread_event = Event() # An event that could potentially be used for signaling.

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this point.
        Functional Utility: Decrements the `count_threads` counter. If it's the last thread
        to arrive, it notifies all other waiting threads and resets the counter. Otherwise,
        it waits.
        """
        
        self.cond.acquire() # Acquire the condition lock.
        Barrier.count_threads -= 1 # Decrement the count of threads that still need to arrive.

        if Barrier.count_threads == 0: # Block Logic: If this is the last thread to reach the barrier.
            self.cond.notify_all() # Notify all other threads waiting on this condition.
            Barrier.count_threads = Barrier.num_threads # Reset the counter for the next use of the barrier.
        else: # Block Logic: If not the last thread, wait for others.
            self.cond.wait() # Release the lock and block until notified.

        self.cond.release() # Release the condition lock.

    @staticmethod
    def add_thread():
        """
        @brief Increments the total number of threads that the barrier should synchronize.
        Functional Utility: This static method is used to dynamically register a new thread
        with the barrier, ensuring the barrier knows how many participants to wait for.
        """
        
        Barrier.num_threads += 1 # Increment total thread count.
        Barrier.count_threads = Barrier.num_threads # Reset current count, effectively preparing for a new synchronization cycle.


class Device(object):
    """
    @class Device
    @brief Represents a single computational device in a distributed system simulation.
    Functional Utility: Manages local sensor data, orchestrates script execution via
    its `DeviceThread`, and coordinates with a supervisor and other devices. It uses
    a shared `Barrier` for synchronization across all devices.
    """
    
    barrier = Barrier() # Class variable: A single global Barrier instance shared by all Device objects.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id (int): A unique identifier for this device.
        @param sensor_data (dict): A dictionary representing sensor data local to this device.
                                  Keys are location IDs, values are data.
        @param supervisor (object): A reference to the supervisor object responsible for
                                    global coordination, such as retrieving neighbour information.
        """
        
        Device.barrier.add_thread() # Register this device's thread with the global barrier.
        self.device_id = device_id # Unique identifier for this device.
        self.sensor_data = sensor_data # Dictionary storing local sensor data.
        self.supervisor = supervisor # Reference to the supervisor object.
        self.script_received = Event() # Event to signal when scripts have been assigned for a timepoint.
        self.scripts = [] # List of (script, location) tuples assigned to this device for execution.
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
        @brief Placeholder for device-specific setup logic.
        Functional Utility: This method is intended to be called during system
        initialization to configure inter-device communication or shared resources.
        In this specific implementation, it currently does nothing (`pass`).
        @param devices (list): A list of all `Device` instances in the simulation.
        """
        
        
        pass # No explicit setup logic implemented in this version.

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
            self.script_received.set() # Set the event to unblock the `DeviceThread`.

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
        ensuring a clean termination of the device's main operations. This typically
        implies that the thread has received a signal to break its main loop.
        """
        
        self.thread.join() # Wait for the main thread to finish.


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief The main worker thread for a `Device`.
    Functional Utility: This thread runs continuously, coordinating the activities
    of its parent `Device`. It fetches global state (neighbours), waits for script
    assignments, and then executes these scripts, collecting data from local
    and neighbouring devices and propagating results.
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
        by collecting data from local and neighbouring devices and updating results.
        """

        while True: # Block Logic: Main loop for continuous timepoint processing until shutdown.
            
            # Functional Utility: Fetch updated neighbour information from the supervisor.
            # Precondition: `self.device.supervisor` is a valid object with a `get_neighbours` method.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Block Logic: Check for shutdown signal (neighbours being None).
                break # Exit the loop and terminate the thread if shutdown is signaled.
            
            Device.barrier.wait() # Synchronization Point: Wait for all device threads to reach this point.
            self.device.script_received.wait() # Synchronization Point: Wait for scripts for the timepoint to be assigned.
            self.device.script_received.clear() # Clear the event for the next timepoint.

            
            # Block Logic: Iterate through assigned scripts and execute them.
            # Invariant: Each script is processed by this thread sequentially for the current timepoint.
            for (script, location) in self.device.scripts:
                script_data = [] # List to accumulate data for the script.
                
                # Block Logic: Collect data from neighbouring devices for the current location.
                for device in neighbours:
                    data = device.get_data(location) # Retrieve data from neighbour.
                    if data is not None:
                        script_data.append(data) # Add data if available.
                
                # Block Logic: Collect data from the current device for the current location.
                data = self.device.get_data(location) # Retrieve local data.
                if data is not None:
                    script_data.append(data) # Add local data if available.

                if script_data != []: # Block Logic: If any data was collected, run the script and update results.
                    
                    # Functional Utility: Execute the script's `run` method with the collected data.
                    result = script.run(script_data)

                    // Functional Utility: Propagate the script's result to neighbouring devices and the current device.
                    for device in neighbours: // Update neighbour data.
                        device.set_data(location, result)
                    self.device.set_data(location, result) // Update local sensor data with the result.