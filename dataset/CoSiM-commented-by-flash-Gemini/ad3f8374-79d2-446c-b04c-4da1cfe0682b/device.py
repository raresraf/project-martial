


"""
@brief This module defines a custom ReusableBarrier class using semaphores, along with Device and DeviceThread classes
for simulating a distributed system.
@details Devices execute scripts, interact with neighbors, and manage shared resources with explicit synchronization,
including a two-phase barrier implementation and a shared lock for data consistency during script execution.
"""

from threading import Lock, Semaphore, Event, Thread

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads using a two-phase semaphore-based approach.
    @details This barrier ensures that all participating threads reach a synchronization point before any are allowed
    to proceed. It uses two phases to allow for reusability without deadlocks.
    @algorithm Two-phase semaphore-based barrier (double-barrier pattern).
    @time_complexity O(N) for `wait` operation due to loop-based semaphore releases, where N is `num_threads`.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes a new instance of the ReusableBarrier.
        @param num_threads (int): The total number of threads that must reach the barrier before it can be passed.
        """
        self.num_threads = num_threads # Total number of threads expected.
        self.count_threads1 = [self.num_threads] # Counter for the first phase. Wrapped in list for pass-by-reference.
        self.count_threads2 = [self.num_threads] # Counter for the second phase. Wrapped in list for pass-by-reference.
        self.count_lock = Lock() # Lock to protect access to the thread counters.
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase. Initially locked.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase. Initially locked.

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all `num_threads` threads have arrived.
        @details This method orchestrates the two phases of the barrier, ensuring reusability.
        @block_logic Manages the two-phase synchronization for barrier reusability.
        @pre_condition `self.count_threads1`, `self.count_threads2`, `self.threads_sem1`, `self.threads_sem2`,
                       and `self.count_lock` are initialized correctly.
        @invariant All threads will be released together after both phases are complete.
        """
        self.phase(self.count_threads1, self.threads_sem1) # First synchronization phase.
        self.phase(self.count_threads2, self.threads_sem2) # Second synchronization phase for reusability.

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the two-phase barrier synchronization.
        @details Threads decrement a shared counter. The last thread to decrement releases all
        waiting threads via a semaphore. The counter is then reset for the next use.
        @param count_threads (list): A list containing the shared counter for this phase.
        @param threads_sem (Semaphore): The semaphore associated with this phase.
        @block_logic Synchronizes threads for a single phase of the barrier.
        @pre_condition `count_threads` is a list with a single integer, `threads_sem` is a Semaphore,
                       and `self.count_lock` is available.
        @invariant All threads waiting on `threads_sem` are released once the counter reaches zero,
                   and the counter is reset.
        """
        with self.count_lock: # Acquire lock to safely modify the shared counter.
            count_threads[0] -= 1 # Decrement the counter for the current phase.
            if count_threads[0] == 0: # Check if this is the last thread in this phase.
                i = 0
                while i < self.num_threads: # Release all `num_threads` waiting threads.
                    threads_sem.release() 
                    i += 1                
                count_threads[0] = self.num_threads # Reset the counter for this phase for reusability.
        threads_sem.acquire() # Acquire the semaphore, waiting if not yet released by the last thread.


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class models an individual device that manages its unique ID, sensor data,
    and interactions with a supervisor. It can receive and execute scripts, which may involve
    modifying its own sensor data and communicating with neighboring devices. Synchronization
    across devices is managed by a shared `ReusableBarrier`, and access to shared resources
    during script execution is protected by a common `Lock`.
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local
    data processing and communication with peers, implementing explicit synchronization
    and locking mechanisms for robust concurrent operation.
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
        self.barrier = None          # Reference to the shared ReusableBarrier for inter-device synchronization.
        self.lock = None             # Reference to the shared Lock for protecting shared resources (e.g., sensor data updates).
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts have been assigned.
        self.scripts = []            # List to store assigned scripts and their locations.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's script assignment.
        self.thread = DeviceThread(self) # The main worker thread for this device.
        self.thread.start()          # Start the device's execution thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives (barrier and lock) for all devices.
        @details This method ensures that a single `ReusableBarrier` and `Lock` instance
        are created by the first device (device 0) and then shared among all other devices
        in the simulation. This centralizes the management of these shared resources.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of shared synchronization
                     and mutual exclusion primitives.
        @pre_condition `devices` is a list of `Device` instances, and this method is called
                       for each device during setup.
        @invariant `self.barrier` and `self.lock` refer to globally shared instances after setup,
                   ensuring consistent synchronization across all devices.
        """
        # Block Logic: Initialize shared barrier and lock only once by the first device (device 0).
        # Pre-condition: `devices[0].barrier` is initially None.
        # Invariant: `devices[0].barrier` and `devices[0].lock` become initialized and shared.
        if devices[0].barrier is None: # Check if the shared barrier has already been initialized.
            if self.device_id == devices[0].device_id: # Only the very first device (device 0) performs initialization.
                bariera = ReusableBarrier(len(devices)) # Create a new reusable barrier.
                my_lock = Lock() # Create a shared lock.
                for device in devices: # Distribute the newly created barrier and lock to all devices.
                    device.barrier = bariera
                    device.lock = my_lock

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
        It's assumed that external synchronization (e.g., through `DeviceThread`'s locking)
        protects this operation during concurrent modifications.
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
    @details This thread orchestrates the device's operational cycle, including
    synchronization via a shared barrier, script execution, and data management.
    It continuously interacts with the supervisor to get neighbor information,
    processes assigned scripts, and ensures proper synchronization using a shared lock
    and the `ReusableBarrier`.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    ensuring proper coordination and data consistency across the distributed system.
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
        it retrieves neighbor information, waits for script assignment completion from the supervisor,
        and then processes the assigned scripts. During script execution, it acquires a shared lock
        to ensure thread-safe access to shared data before performing updates. After processing all
        scripts, it clears the `timepoint_done` event and synchronizes with other `DeviceThread`
        instances via the shared `ReusableBarrier`. The loop terminates when the supervisor signals
        the end of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        script execution with mutual exclusion, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `timepoint_done` event, `lock`, and `barrier`.
        @invariant The thread progresses through timepoints, processes scripts under lock protection,
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

            # Block Logic: Iterate through all assigned scripts for the current timepoint and execute them.
            # Pre-condition: `self.device.scripts` contains tuples of (script, location).
            # Invariant: Each script is run with collected data and results are propagated to neighbors and itself.
            for (script, location) in self.device.scripts:
                # Block Logic: Acquire the shared lock to protect access to sensor data during script execution.
                # Invariant: Only one device's script can modify shared sensor data at a time.
                self.device.lock.acquire()
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
                self.device.lock.release() # Release the shared lock after processing the script.
            
            # Functional Utility: Clear the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Functional Utility: Clear the scripts list for the next timepoint.
            self.device.scripts = [] # Reset scripts list for the next timepoint.
            
            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances will reach this barrier before any proceeds to the next timepoint.
            self.device.barrier.wait()
