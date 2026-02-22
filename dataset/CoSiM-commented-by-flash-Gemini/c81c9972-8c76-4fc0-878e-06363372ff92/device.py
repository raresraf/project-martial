

"""
@brief This module defines `Device`, `DeviceThread`, `ScriptThread` classes, and a custom `ReusableBarrier`
for simulating a distributed system.
@details It features a condition-variable-based reusable barrier for inter-device synchronization.
The main `DeviceThread` manages and distributes scripts to a pool of worker `ScriptThread`s,
which execute scripts and store results. These results are then applied by the main `DeviceThread`,
utilizing locks for thread-safe data access and coordination.
"""

from threading import Condition, Event, Lock, Thread


class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads using a condition variable.
    @details This barrier allows a set number of threads to wait until all participants arrive at a synchronization point,
    after which all threads are released simultaneously. The barrier can then be reset for subsequent synchronization points.
    It also includes a `reinit` method to dynamically adjust the number of participating threads.
    @algorithm Condition Variable based synchronization.
    @time_complexity O(1) for `wait` operation, assuming constant time for underlying threading primitives.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes a new instance of the ReusableBarrier.
        @param num_threads (int): The total number of threads that must reach the barrier before it can be passed.
        """
        self.num_threads = num_threads # Total number of threads expected.
        self.count_threads = self.num_threads # Current count of threads waiting at the barrier.
        self.cond = Condition() # Condition variable used for thread synchronization.

    def reinit(self):
        """
        @brief Reinitializes the barrier for a potentially new number of threads and forces a wait.
        @details This method is used to signal a barrier re-configuration, typically during a termination sequence.
        It decrements the `num_threads` count and then calls `wait()`, effectively acting as a mechanism to
        allow a dynamic reduction in participating threads, often used for graceful shutdown.
        @block_logic Dynamically adjusts the barrier's expected thread count and initiates a wait cycle.
        @pre_condition `self.cond` is an initialized Condition object, `self.num_threads` is positive.
        @invariant `self.num_threads` is decremented by one, and a `wait` operation is performed.
        """
        self.cond.acquire() # Acquire the lock associated with the condition variable.
        self.num_threads -= 1 # Decrement the total number of threads, preparing for a reduced count.
        self.cond.release() # Release the lock.
        self.wait() # Force the current thread to wait at the barrier.

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all `num_threads` threads have arrived.
        @details Once all threads have arrived, they are all released, and the barrier is reset for future use.
        @block_logic Thread synchronization mechanism.
        @pre_condition `self.cond` is an initialized Condition object, `self.count_threads` accurately reflects
                       the number of threads currently waiting or yet to arrive.
        @invariant All threads attempting to pass the barrier will eventually be released together.
        """
        self.cond.acquire() # Acquire the lock associated with the condition variable.
        self.count_threads -= 1 # Decrement the count of threads yet to arrive.
        if self.count_threads == 0: # Check if this is the last thread to arrive at the barrier.
            self.cond.notify_all() # Release all waiting threads.
            self.count_threads = self.num_threads # Reset the thread count, making the barrier reusable.
        else:
            self.cond.wait() # Wait for other threads to arrive (releases the lock implicitly).
        self.cond.release() # Release the lock after being notified or decrementing the count.


"""
@brief This module defines `Device`, `DeviceThread`, `ScriptThread` classes, and a custom `ReusableBarrier`
for simulating a distributed system.
@details It features a condition-variable-based reusable barrier for inter-device synchronization.
The main `DeviceThread` manages and distributes scripts to a pool of worker `ScriptThread`s,
which execute scripts and store results. These results are then applied by the main `DeviceThread`,
utilizing locks for thread-safe data access and coordination.
"""

from threading import Condition, Event, Lock, Thread


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It orchestrates script execution by coordinating its dedicated `DeviceThread` which, in turn,
    manages a pool of `ScriptThread`s. Synchronization across devices is managed by a shared
    `ReusableBarrier`. Access to shared resources like `locations_lock`, `results`, and `results_lock`
    is protected to ensure data consistency.
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, implementing explicit synchronization and mutual exclusion
    for coordinated and consistent execution across timepoints, especially with complex, multi-stage
    script processing and result aggregation.
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
        self.devices = []            # Reference to the list of all Device objects in the simulation.
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are ready for execution.
        self.start = Event()         # Event to signal the DeviceThread to start its main loop (after setup).
        self.scripts = []            # List to store assigned scripts and their locations (temporarily).
        self.locations_lock = {}     # Dictionary to hold locks for each sensor data location (shared across devices).

        self.scripts_to_process = [] # List of scripts ready to be processed by ScriptThreads.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's script assignment.
        self.nr_script_threats = 0   # Counter for currently active ScriptThread instances.
        self.thread = DeviceThread(self) # The main controlling thread for this device.
        self.thread.start()          # Start the device's main controlling thread.
        self.script_threats = []     # List to hold references to active ScriptThread instances.
        self.barrier_devices = None  # Reference to the shared ReusableBarrier for inter-device synchronization.
        self.neighbours = None       # Stores the list of neighboring devices for the current timepoint.
        self.cors = 8                # Constant: number of concurrent ScriptThread instances to run in a batch.
        self.lock = None             # Reference to a shared Lock for general resource protection across devices.
        self.results = {}            # Dictionary to store results from ScriptThread instances (keyed by location).
        self.results_lock = None     # Reference to a shared Lock for protecting `results` dictionary.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives (barrier and global locks) for all devices.
        @details This method ensures that a single `ReusableBarrier` instance (`barrier`), a shared
        `Lock` for general resource protection (`lock`), and a shared `Lock` for results (`results_lock`)
        are created only once by device 0 and then distributed to all other devices in the simulation.
        It also populates the `devices` list for all devices and signals their `DeviceThread`s to start.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of shared synchronization
                     and mutual exclusion primitives, and signaling device thread readiness.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `self.barrier_devices`, `self.lock`, and `self.results_lock` refer to globally shared
                   instances after setup. `self.devices` holds references to all simulation devices.
                   All `DeviceThread`s are signaled to start.
        """
        if self.device_id == 0: # Only device 0 is responsible for initializing and distributing shared resources.
            global_lock = Lock() # Create a shared lock for general resource protection.
            global_results_lock = Lock() # Create a shared lock for results dictionary.
            global_barrier = ReusableBarrier(len(devices)) # Create a reusable barrier, sized for all devices.
            # Block Logic: Distribute the newly created shared resources to all devices.
            # Invariant: Each device in `devices` receives a reference to the shared locks and barrier.
            for device in devices:
                device.lock = global_lock # Assign the shared general lock.
                device.results_lock = global_results_lock # Assign the shared results lock.
                device.barrier_devices = global_barrier # Assign the shared barrier.
                device.devices = devices # Assign the full list of devices.
            # Block Logic: Signal all DeviceThreads to start their main loop.
            # Invariant: `device.start` event is set for each device.
            for device in devices:
                device.start.set()

        # No explicit handling for `scripts_to_process` based on `self.scripts` here.
        # This implies `scripts_to_process` is populated later or based on scripts already assigned.
        # However, the loop `for script in self.scripts: self.scripts_to_process.append(script)`
        # in DeviceThread.run is problematic as it re-adds scripts already assigned.
        # Assuming `self.scripts_to_process` is intended to be the queue of new scripts.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location, and ensures a lock exists for it.
        @details This method uses the shared `self.lock` to ensure thread-safe creation of location-specific locks.
        For a given `location`, it checks if a lock already exists in any `Device`'s `locations_lock` dictionary
        across the simulation. If not, it creates a new `Lock` for it and assigns it to all devices.
        If a `script` is provided, it's added to the device's `scripts` and `scripts_to_process` lists,
        and `script_received` is set. If `script` is None, `timepoint_done` and `script_received` are set.
        @param script (object): The script object to be executed, or None to signal end of assignments.
        @param location (str): The location associated with the script or data.
        @block_logic Handles script assignment and dynamically ensures existence of a location-specific lock.
        @pre_condition `self.scripts` and `self.scripts_to_process` are lists, `self.script_received` and
                       `self.timepoint_done` are Event objects. `self.lock` and `self.devices` are initialized.
        @invariant A script is added to lists and `script_received` is set, or `timepoint_done` is set.
                   A lock for `location` is ensured to exist in `locations_lock` across all devices.
        """
        # Block Logic: Acquire the global lock to protect modifications to `locations_lock` across devices.
        self.lock.acquire()
        # Block Logic: Ensure a lock exists for `location` in all devices' `locations_lock` dictionaries.
        # Invariant: `device.locations_lock[location]` holds a valid Lock object for all devices.
        for device in self.devices: # Iterate through all devices in the simulation.
            if location not in device.locations_lock.keys(): # Check if this location has a lock assigned yet.
                device.locations_lock[location] = Lock() # Create a new lock for this location if it doesn't exist.
        self.lock.release() # Release the global lock.

        if script is not None:
            self.scripts.append((script, location)) # Add the script to the main scripts list.
            self.scripts_to_process.append((script, location)) # Add to the list to be processed by threads.
            self.script_received.set() # Signal that new scripts have been received.
        else:
            self.timepoint_done.set() # Signal that script assignments for the current timepoint are complete.
            self.script_received.set() # Also signal script received, as the loop might be waiting on it.
            
    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location (str): The location for which to retrieve data.
        @return object: The sensor data at the specified location, or None if the location is not found.
        """
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else:
            data = None
        return data
        
    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @details This method updates the internal sensor data if the location exists.
        It's assumed that external synchronization (e.g., through `DeviceThread`'s or `ScriptThread`'s locks)
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
        @brief Shuts down the device by joining its associated controlling thread.
        @details This ensures that the device's main `DeviceThread` completes its execution before the program exits.
        """
        self.thread.join()



class DeviceThread(Thread):
    """
    @brief The main controlling thread for a Device instance.
    @details This thread orchestrates the device's operational cycle for each timepoint.
    It manages the fetching of neighbor information, processes assigned scripts by spawning
    `ScriptThread`s in batches, aggregates their results, and applies these results to the
    sensor data, while coordinating with the global barrier.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    abstracting multi-threaded script execution, facilitating result aggregation, and ensuring proper
    coordination and data consistency within the distributed system.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
        self.device = device # Reference to the parent Device object.
        self.device.neighbours = None # Initialize neighbours to None.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        @details This method first waits for the initial setup to complete (`self.device.start`).
        It then enters a continuous loop where it fetches neighbor information from the supervisor.
        If neighbours are available, it processes scripts from `self.device.scripts_to_process`
        in batches up to `self.device.cors` (8). For each batch, it spawns `ScriptThread`s,
        waits for their completion, and then applies the aggregated results to sensor data,
        protected by location-specific locks. After all scripts for the timepoint are processed,
        it clears `timepoint_done` and synchronizes with other `DeviceThread` instances via the
        global `ReusableBarrier`. The loop terminates when the supervisor signals the end
        of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        batched parallel script execution, result aggregation, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `scripts_to_process` list, `timepoint_done` and `script_received` events,
                       `nr_script_threats`, `barrier_devices`, `neighbours`, `cors`, `lock`, `results`, and `results_lock`.
        @invariant The thread progresses through timepoints, processes scripts concurrently in batches,
                   aggregates results, applies updates, and ensures global synchronization.
        """
        self.device.start.wait() # Wait until setup is complete and the Device is ready to operate.
        while True:
            # Functional Utility: Get information about neighboring devices from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Check if the simulation should terminate.
            # Pre-condition: `self.device.neighbours` list indicates the current state of the network.
            # Invariant: The loop terminates if no neighbors are returned by the supervisor.
            if self.device.neighbours is None:
                self.device.barrier_devices.reinit() # Call reinit on barrier, likely to allow threads to exit.
                break

            # Functional Utility: Clear scripts from previous timepoint for fresh processing.
            self.device.scripts_to_process = []
            for script in self.device.scripts: # Repopulate `scripts_to_process` from `self.device.scripts`.
                self.device.scripts_to_process.append(script)

            # Functional Utility: Clear results from previous timepoint.
            self.device.results = {}

            # Block Logic: Main loop for processing scripts for the current timepoint.
            # Invariant: Continues until all scripts for the timepoint are processed and `timepoint_done` is set.
            while True:
                # Block Logic: Wait for new scripts to be assigned or for timepoint completion.
                # Invariant: `self.device.script_received` is set when new scripts are available.
                if not self.device.timepoint_done.is_set():
                    self.device.script_received.wait() # Wait for supervisor to assign scripts.
                    self.device.script_received.clear() # Clear event for next cycle.

                # Block Logic: Check if all scripts are processed for the current timepoint.
                # Invariant: If `scripts_to_process` is empty and `timepoint_done` is set, exit processing loop.
                if len(self.device.scripts_to_process) == 0:
                    if self.device.timepoint_done.is_set():
                        break # All scripts processed for this timepoint.

                # Block Logic: Process scripts in batches using `ScriptThread` workers.
                while len(self.device.scripts_to_process):
                    list_threats = [] # Temporary list to hold scripts for the current batch.
                    self.device.script_threats = [] # List to hold ScriptThread instances.
                    self.device.nr_script_threats = 0 # Reset active ScriptThread counter.
                    
                    # Block Logic: Populate `list_threats` with scripts up to `self.device.cors`.
                    # Invariant: `list_threats` contains a batch of scripts for concurrent execution.
                    while len(self.device.scripts_to_process) and self.device.nr_script_threats < self.device.cors:
                        script, location = self.device.scripts_to_process.pop(0) # Get next script.
                        list_threats.append((script, location))
                        self.device.nr_script_threats += 1 # Increment active ScriptThread counter.

                    # Block Logic: For each script in the current batch, create data, spawn a ScriptThread, and start it.
                    # Invariant: `ScriptThread`s are created and started for each script in the batch.
                    for script, location in list_threats:
                        script_data = [] # Data for the current script.
                        
                        neighbours = self.device.neighbours # Reference to current neighbours.
                        # Block Logic: Collect data from neighboring devices for the current location.
                        for device in neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        # Block Logic: Collect data from the current device itself for the current location.
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        # Functional Utility: Create and start a new ScriptThread.
                        thread_script_d = ScriptThread(self.device, script, location, script_data)
                        self.device.script_threats.append(thread_script_d)
                        thread_script_d.start()

                    # Block Logic: Wait for all ScriptThread instances in the current batch to complete.
                    # Invariant: The DeviceThread waits until all `ScriptThread`s in the batch finish.
                    for thread in self.device.script_threats:
                        thread.join()

            # Block Logic: Apply aggregated results from ScriptThreads to sensor data.
            # Invariant: `self.device.sensor_data` is updated with all script results for the timepoint.
            for location, result in self.device.results.iteritems(): # Using iteritems (Python 2) for dictionary iteration.
                # Block Logic: Acquire location-specific lock to ensure thread-safe update.
                self.device.locations_lock[location].acquire()
                # Block Logic: Propagate the script's result to neighboring devices.
                for device in self.device.neighbours:
                    device.set_data(location, result)
                # Functional Utility: Update the current device's own data.
                self.device.set_data(location, result)
                self.device.locations_lock[location].release() # Release location-specific lock.

            # Functional Utility: Clear events for the next timepoint.
            self.device.timepoint_done.wait() # Wait for final timepoint_done confirmation.
            self.device.timepoint_done.clear() # Clear event for next timepoint.

            # Functional Utility: Clear the scripts list for the next timepoint.
            self.device.scripts = [] # Reset scripts list for the next timepoint.

            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances in the simulation will reach this barrier before any proceeds.
            self.device.barrier_devices.wait()


class ScriptThread(Thread):
    """
    @brief A worker thread dedicated to executing a single assigned script.
    @details This thread collects necessary input data for its script, executes the script,
    and stores its result in the parent `Device`'s shared `results` dictionary. It does not
    directly update sensor data but delegates this to the `DeviceThread` after aggregation.
    @architectural_intent Enhances parallelism by offloading individual script execution
    from the main `DeviceThread`, allowing for concurrent processing while centralizing result aggregation.
    """
    
    def __init__(self, device, script, location, script_data):
        """
        @brief Initializes a new ScriptThread instance.
        @param device (Device): The parent Device object that this script thread serves.
        @param script (object): The script object to be executed.
        @param location (str): The location associated with the script for which data is processed.
        @param script_data (list): The collected input data required for the script's execution.
        """
        Thread.__init__(self, name="Device Script Thread %d" % device.device_id) # Initialize base Thread.
        self.device = device # Reference to the parent Device.
        self.location = location # The sensor data location this script pertains to.
        self.script = script # The script to execute.
        self.script_data = script_data # Pre-collected data for the script.

    def run(self):
        """
        @brief The main execution logic for the ScriptThread.
        @details This method executes the assigned script with the provided `script_data`.
        If the script returns a result, it stores this result in the parent `Device`'s
        `results` dictionary, protected by `self.device.results_lock`. It also decrements
        the `self.device.nr_script_threats` counter.
        @block_logic Executes a script and stores its result in a shared, thread-safe manner.
        @pre_condition `self.script` is an object with a `run` method, `self.script_data` is a list.
                       `self.device.results` is a dictionary, `self.device.results_lock` is a Lock.
        @invariant `self.device.results` may be updated with the script's output, and `nr_script_threats` is decremented.
        """
        if self.script_data != []:
            result = self.script.run(self.script_data) # Execute the script.
            
            # Block Logic: Store the result in the shared `device.results` dictionary, protected by a lock.
            self.device.results_lock.acquire()
            self.device.results[self.location] = result # Store the result for the specific location.
            self.device.results_lock.release()
        self.device.nr_script_threats -= 1 # Decrement the counter for active script threads.