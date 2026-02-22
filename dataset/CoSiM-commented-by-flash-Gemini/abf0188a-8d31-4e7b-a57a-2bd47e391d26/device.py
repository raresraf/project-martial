

"""
@brief This module defines `Device`, `DeviceThread`, and `MyReusableBarrier` classes for simulating a distributed system.
@details It implements a custom two-phase semaphore-based reusable barrier (`MyReusableBarrier`) for inter-device
synchronization, where each `Device` executes scripts sequentially within its dedicated `DeviceThread`. This design
facilitates coordinated execution and data consistency across multiple simulated devices.
"""

from threading import * # Import all threading components directly.


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It can receive and queue scripts for execution, which are then processed sequentially by its
    dedicated `DeviceThread`. Synchronization across all `Device` instances in the simulation
    is managed by a shared `MyReusableBarrier`.
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, implementing explicit two-phase barrier synchronization
    for coordinated execution across timepoints.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary containing initial sensor data,
                                   where keys are locations and values are data readings.
        @param supervisor (object): A reference to the supervisor object that manages
                                    the overall distributed system and provides access
                                    to network information (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are ready for execution.
        self.scripts = []            # List to store assigned scripts and their locations.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's script assignment.
        self.thread = DeviceThread(self) # The main worker thread for this device.
        self.thread.start()          # Start the device's execution thread.
        self.devices = []            # Reference to the list of all Device objects in the simulation.
        self.bar = None              # Reference to the shared MyReusableBarrier for inter-device synchronization.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared `MyReusableBarrier` for all devices.
        @details This method initializes a single `MyReusableBarrier` instance (only by the first device
        in the `devices` list) and distributes it to all other devices in the simulation.
        It stores a reference to all devices and assigns the shared barrier to each of them.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of the shared barrier.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `self.bar` refers to a globally shared `MyReusableBarrier` instance after setup.
                   `self.devices` holds references to all simulation devices.
        """
        self.devices = devices # Store the list of all devices.
        # Block Logic: Initialize the shared barrier only once by the first device (devices[0]).
        # Invariant: The barrier is created and then distributed to all devices.
        if self == devices[0]: # Check if this is the first device in the list.
            self.bar = MyReusableBarrier(len(devices)) # Create a new reusable barrier, sized for all devices.
        # Note: Other devices implicitly get the barrier reference from devices[0] in DeviceThread.run().

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location.
        @details If a script is provided, it's appended to the device's script queue, and the
        `script_received` event is set. If no script is provided (i.e., `script` is None),
        it signifies that the current timepoint's script assignment is complete, and the
        `timepoint_done` event is set to unblock the `DeviceThread`.
        @param script (object): The script object to be executed, or None to signal end of assignments.
        @param location (str): The location associated with the script or data.
        @block_logic Handles the assignment of new scripts or signals the completion of script assignment for a timepoint.
        @pre_condition `self.scripts` is a list, `self.script_received` and `self.timepoint_done` are Event objects.
        @invariant Either a script is added and `script_received` is set, or `timepoint_done` is set.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the queue.
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # Signal that script assignments for the current timepoint are complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location (str): The location for which to retrieve data.
        @return object: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @details This method updates the internal sensor data if the location exists.
        @param location (str): The location whose data is to be updated.
        @param data (object): The new data value for the specified location.
        @block_logic Updates the internal sensor data.
        @pre_condition `self.sensor_data` is a dictionary.
        @invariant If `location` is a key in `self.sensor_data`, its value is updated.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its associated thread.
        @details This ensures that the device's worker thread completes its execution before the program exits.
        """
        self.thread.join()



class DeviceThread(Thread):
    """
    @brief The main worker thread for a Device instance.
    @details This thread orchestrates the device's operational cycle for each timepoint.
    It fetches neighbor information, waits for script assignments, and then processes
    these scripts sequentially. It ensures inter-device synchronization using a shared
    `MyReusableBarrier`.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    ensuring proper coordination and data consistency within the distributed system
    through sequential script execution and barrier synchronization.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
        self.device = device # Reference to the parent Device object.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        @details This method continuously monitors the simulation state. For each timepoint,
        it retrieves neighbor information from the supervisor. If neighbors are available,
        it waits until `timepoint_done` is set (signaling that script assignments are complete),
        then sequentially processes each assigned script. For each script, it collects data
        from neighbors and its own sensors, executes the script, and propagates results back
        to neighbors and itself. After all scripts for the timepoint are processed, it clears
        `timepoint_done` and synchronizes with other `DeviceThread` instances via the global
        `MyReusableBarrier`. The loop terminates when the supervisor signals the end
        of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        sequential script execution, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `scripts` list, `timepoint_done` event, and `bar` (MyReusableBarrier).
        @invariant The thread progresses through timepoints, processes scripts sequentially,
                   and ensures global synchronization.
        """
        while True:
            # Functional Utility: Get information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # Block Logic: Check if the simulation should terminate.
            # Pre-condition: `neighbours` list indicates the current state of the network.
            # Invariant: The loop terminates if no neighbors are returned by the supervisor.
            if neighbours is None:
                break

            # Block Logic: Wait until script assignments for the current timepoint are complete.
            # Pre-condition: `self.device.timepoint_done` is an Event object.
            # Invariant: The thread proceeds only after the supervisor signals completion of script assignment.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Clear the event for the next timepoint.

            # Block Logic: Iterate through all assigned scripts for the current timepoint and execute them sequentially.
            # Pre-condition: `self.device.scripts` contains tuples of (script, location).
            # Invariant: Each script is run with collected data and results are propagated to neighbors and itself.
            for (script, location) in self.device.scripts:
                script_data = [] # List to accumulate data for the current script's execution.
                
                # Block Logic: Collect data from neighboring devices for the current location.
                # Invariant: `script_data` will contain data from all available neighbors for the given location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collect data from the current device itself for the current location.
                # Invariant: If available, the device's own data for the location is added to `script_data`.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Execute the script if there is any data to process.
                # Pre-condition: `script` is an object with a `run` method, and `script_data` is a list of data.
                # Invariant: `result` holds the output of the script's execution.
                if script_data != []:
                    result = script.run(script_data) # Execute the script with the collected data.

                    # Block Logic: Propagate the script's result to neighboring devices.
                    # Invariant: All neighbors receive the updated data for the given location.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Functional Utility: Update the current device's own data with the script's result.
                    self.device.set_data(location, result)

            # Functional Utility: Clear the scripts list for the next timepoint.
            self.device.scripts = [] # Reset scripts list for the next timepoint.
            
            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Pre-condition: `self.device.devices[0].bar` is an initialized `MyReusableBarrier`.
            # Invariant: All DeviceThread instances in the simulation will reach this barrier before any proceeds to the next timepoint.
            self.device.devices[0].bar.wait() # Access the barrier through the first device in the list.


class MyReusableBarrier():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads using a two-phase semaphore-based approach.
    @details This barrier ensures that all participating threads reach a synchronization point before any are allowed
    to proceed. It uses two distinct phases with semaphores and a shared counter to achieve reusability without deadlocks.
    @algorithm Two-phase semaphore-based barrier (double-barrier pattern).
    @time_complexity O(N) for `wait` operation due to loop-based semaphore releases, where N is `num_threads`.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes a new instance of the MyReusableBarrier.
        @param num_threads (int): The total number of threads that must reach the barrier before it can be passed.
        """
        self.num_threads = num_threads # Total number of threads expected.
        self.count_threads1 = self.num_threads # Counter for the first phase.
        self.count_threads2 = self.num_threads # Counter for the second phase.
        
        self.counter_lock = Lock()       # Lock to protect access to the thread counters.
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase. Initially locked.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase. Initially locked.

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all `num_threads` threads have arrived.
        @details This method orchestrates the two phases of the barrier, ensuring reusability.
        @block_logic Manages the two-phase synchronization for barrier reusability.
        @pre_condition `self.count_threads1`, `self.count_threads2`, `self.threads_sem1`, `self.threads_sem2`,
                       and `self.counter_lock` are initialized correctly.
        @invariant All threads will be released together after both phases are complete.
        """
        self.phase1() # First synchronization phase.
        self.phase2() # Second synchronization phase for reusability.

    def phase1(self):
        """
        @brief Implements the first phase of the two-phase barrier synchronization.
        @details Threads decrement a shared counter. The last thread to decrement releases all
        waiting threads via `threads_sem1`. The counter for `phase2` is also reset here.
        @block_logic Synchronizes threads for the first phase of the barrier.
        @pre_condition `self.count_threads1` is an integer, `self.threads_sem1` is a Semaphore,
                       and `self.counter_lock` is available.
        @invariant All threads waiting on `threads_sem1` are released once `count_threads1` reaches zero,
                   and `count_threads2` is reset.
        """
        with self.counter_lock: # Acquire lock to safely modify the shared counter.
            self.count_threads1 -= 1 # Decrement the counter for the first phase.
            if self.count_threads1 == 0: # Check if this is the last thread in this phase.
                for i in range(self.num_threads): # Release all `num_threads` waiting threads.
                    self.threads_sem1.release()
                self.count_threads2 = self.num_threads # Reset counter for the next phase.
         
        self.threads_sem1.acquire() # Acquire the semaphore, waiting if not yet released by the last thread.
         
    def phase2(self):
        """
        @brief Implements the second phase of the two-phase barrier synchronization.
        @details Threads decrement a shared counter. The last thread to decrement releases all
        waiting threads via `threads_sem2`. The counter for `phase1` is also reset here.
        @block_logic Synchronizes threads for the second phase of the barrier.
        @pre_condition `self.count_threads2` is an integer, `self.threads_sem2` is a Semaphore,
                       and `self.counter_lock` is available.
        @invariant All threads waiting on `threads_sem2` are released once `count_threads2` reaches zero,
                   and `count_threads1` is reset.
        """
        with self.counter_lock: # Acquire lock to safely modify the shared counter.
            self.count_threads2 -= 1 # Decrement the counter for the second phase.
            if self.count_threads2 == 0: # Check if this is the last thread in this phase.
                for i in range(self.num_threads): # Release all `num_threads` waiting threads.
                    self.threads_sem2.release()
                self.count_threads1 = self.num_threads # Reset counter for the next phase.
         
        self.threads_sem2.acquire() # Acquire the semaphore, waiting if not yet released by the last thread.

