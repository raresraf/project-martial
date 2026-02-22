

"""
@brief This module defines `Device`, `DeviceThread` classes, and a custom `ReusableBarrierSem`
for simulating a distributed system.
@details It features a semaphore-based reusable barrier for inter-device synchronization, where each `Device`
manages a pool of worker `DeviceThread`s. These worker threads collaboratively process scripts, using a
class-level list of location-specific locks for thread-safe data access, optimizing concurrent data
processing and ensuring data integrity within the simulation.
"""

from threading import Lock, Event, Thread, Semaphore, Condition


class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads using a two-phase semaphore-based approach.
    @details This barrier ensures that all participating threads reach a synchronization point before any are allowed
    to proceed. It uses two phases (`phase1` and `phase2`) to allow for reusability without deadlocks.
    @algorithm Two-phase semaphore-based barrier (double-barrier pattern).
    @time_complexity O(N) for `wait` operation due to loop-based semaphore releases, where N is `num_threads`.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes a new instance of the ReusableBarrierSem.
        @param num_threads (int): The total number of threads that must reach the barrier before it can be passed.
        """
        self.num_threads = num_threads # Total number of threads expected.
        self.count_threads1 = self.num_threads # Counter for the first phase.
        self.count_threads2 = self.num_threads # Counter for the second phase.
        self.counter_lock = Lock() # Lock to protect access to the thread counters.
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
        waiting threads via `threads_sem1`. The counter for `phase1` is also reset here for reusability.
        @block_logic Synchronizes threads for the first phase of the barrier.
        @pre_condition `self.count_threads1` is an integer, `self.threads_sem1` is a Semaphore,
                       and `self.counter_lock` is available.
        @invariant All threads waiting on `threads_sem1` are released once `count_threads1` reaches zero,
                   and `count_threads1` is reset.
        """
        with self.counter_lock: # Acquire lock to safely modify the shared counter.
            self.count_threads1 -= 1 # Decrement the counter for the first phase.
            if self.count_threads1 == 0: # Check if this is the last thread in this phase.
                for i in range(self.num_threads): # Release all `num_threads` waiting threads.
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset counter for the next phase.
        self.threads_sem1.acquire() # Acquire the semaphore, waiting if not yet released by the last thread.

    def phase2(self):
        """
        @brief Implements the second phase of the two-phase barrier synchronization.
        @details Threads decrement a shared counter. The last thread to decrement releases all
        waiting threads via `threads_sem2`. The counter for `phase2` is also reset here for reusability.
        @block_logic Synchronizes threads for the second phase of the barrier.
        @pre_condition `self.count_threads2` is an integer, `self.threads_sem2` is a Semaphore,
                       and `self.counter_lock` is available.
        @invariant All threads waiting on `threads_sem2` are released once `count_threads2` reaches zero,
                   and `count_threads2` is reset.
        """
        with self.counter_lock: # Acquire lock to safely modify the shared counter.
            self.count_threads2 -= 1 # Decrement the counter for the second phase.
            if self.count_threads2 == 0: # Check if this is the last thread in this phase.
                for i in range(self.num_threads): # Release all `num_threads` waiting threads.
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for the next phase.
        self.threads_sem2.acquire() # Acquire the semaphore, waiting if not yet released by the last thread.


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It coordinates a fixed group of `DeviceThread`s to execute assigned scripts in parallel.
    Synchronization across devices is managed by a class-level shared `ReusableBarrierSem` (`barrier`),
    and thread-safe access to per-location sensor data is ensured by a class-level list of
    `(location, Lock)` tuples (`location_locks`). Each device spawns `nr_t` (8) `DeviceThread`s.
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, utilizing an internal pool of worker threads for parallel
    script execution and granular locking for data consistency and integrity.
    """
    
    location_locks = [] # Class-level list of (location, Lock) tuples for global location-specific data protection.
    barrier = None      # Class-level shared ReusableBarrierSem for inter-device synchronization.
    nr_t = 8            # Class-level constant: number of DeviceThread workers per device.

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
        self.neighbours_event = Event() # Event used for synchronizing the fetching of neighbours by internal worker threads.
        self.threads = []            # List to hold the DeviceThread worker instances for this device.
        # Block Logic: Create and start `nr_t` DeviceThread workers for this device.
        # Invariant: `self.threads` contains `nr_t` active `DeviceThread` instances.
        for i in xrange(Device.nr_t): # Using xrange (Python 2 syntax) for loop.
            self.threads.append(DeviceThread(self, i))
        for i in xrange(Device.nr_t):
            self.threads[i].start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the class-level shared `ReusableBarrierSem` for all devices.
        @details This method initializes a single `ReusableBarrierSem` instance for the entire simulation,
        sized to synchronize all `DeviceThread` workers from all devices. This class-level barrier
        is then shared among all devices.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization of the global synchronization barrier.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `Device.barrier` refers to a globally shared `ReusableBarrierSem` instance after setup.
        """
        # Functional Utility: Create the global barrier, sized for all devices' worker threads.
        # This synchronizes all `nr_t` threads from each `Device` instance.
        Device.barrier = ReusableBarrierSem(Device.nr_t * len(devices))

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location, and dynamically manages location locks.
        @details If a script is provided, it's appended to the device's script queue. Before appending,
        it checks if a lock for the `location` already exists in the class-level `Device.location_locks`.
        If not, a new `Lock` is created and added to `Device.location_locks` for that location.
        The `script_received` event is set to signal the `DeviceThread` workers. If no script is provided
        (i.e., `script` is None), it signifies that the current timepoint's script assignment is complete,
        and the `timepoint_done` event is set to unblock the `DeviceThread` workers.
        @param script (object): The script object to be executed, or None to signal end of assignments.
        @param location (str): The location associated with the script or data.
        @block_logic Handles script assignment and dynamically ensures existence of a location-specific lock.
        @pre_condition `self.scripts` is a list, `self.script_received` and `self.timepoint_done` are Event objects.
                       `Device.location_locks` is a shared list for location-specific locks.
        @invariant Either a script is added and `script_received` is set, or `timepoint_done` is set.
                   A lock for `location` is ensured to exist in `Device.location_locks`.
        """
        # Block Logic: Dynamically create a lock for `location` if it doesn't already exist in the shared list.
        # Invariant: `Device.location_locks` contains a (location, Lock) tuple for the given `location`.
        if location not in [elem[0] for elem in Device.location_locks]:
            Device.location_locks.append((location, Lock()))

        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the queue.
            self.script_received.set() # Signal that new scripts have been received.
        else:
            self.timepoint_done.set() # Signal that script assignments for the current timepoint are complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location (str): The location for which to retrieve data.
        @return object: The sensor data at the specified location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @details This method updates the internal sensor data if the location exists.
        It's assumed that external synchronization (e.g., through `DeviceThread`s' `location_locks`)
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
        @brief Shuts down the device by joining its associated worker threads.
        @details This ensures that all `DeviceThread` workers complete their execution before the program exits.
        """
        for i in xrange(Device.nr_t):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    @brief A worker thread managed by a Device instance, designed to collaboratively process scripts.
    @details Multiple `DeviceThread` instances within a single `Device` work together. This thread
    synchronizes with its peers within the same device using a shared event (`neighbours_event`)
    and with all `Device` instances via the class-level `ReusableBarrierSem`. Each thread processes
    a subset of scripts assigned in a round-robin fashion, acquiring location-specific locks
    before accessing shared data.
    @architectural_intent Enables fine-grained parallel processing of scripts within a single `Device`
    by dividing the workload among multiple threads, improving throughput while maintaining synchronization
    and data consistency through managed locks and barriers.
    """

    def __init__(self, device, index):
        """
        @brief Initializes a new DeviceThread worker.
        @param device (Device): The parent Device object that this thread serves.
        @param index (int): A unique index for this specific worker thread within its parent Device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
        self.device = device # Reference to the parent Device object.
        self.index = index   # Unique index of this worker thread within its Device.
        self.neighbours = None # Stores the list of neighboring devices for the current timepoint.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread worker.
        @details This method continuously monitors the simulation state. For each timepoint,
        the worker with `index == 0` fetches neighbor information from the supervisor and sets
        `self.device.neighbours_event` to signal others. All workers then wait for this event
        and retrieve the shared `neighbours` list. If neighbors are available, they wait for
        `timepoint_done` (signaling script assignments are complete). Workers then process their
        assigned scripts (based on `self.index` in a round-robin fashion), acquiring location-specific
        locks for data access. After processing, they synchronize using `Device.barrier`, and `index == 0`
        clears events for the next timepoint. The loop terminates when the supervisor signals
        the end of the simulation.
        @block_logic Manages the collaborative and synchronized execution of scripts within a Device.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `scripts` list, `timepoint_done` event, `neighbours_event`, and `Device.barrier`.
        @invariant The thread progresses through timepoints, processes its assigned scripts,
                   and ensures both intra-device and inter-device synchronization.
        """
        while True:
            # Block Logic: Only the worker thread with index 0 fetches neighbor information from the supervisor.
            # Other worker threads wait for this event to be set.
            if self.index == 0:
                self.neighbours = self.device.supervisor.get_neighbours() # Functional Utility: Get neighbours.
                self.device.neighbours_event.set() # Signal other worker threads that neighbours are fetched.
            else:
                self.device.neighbours_event.wait() # Wait for `index == 0` thread to fetch neighbours.
                self.neighbours = self.device.threads[0].neighbours # Retrieve neighbours from the first worker thread.
            
            # Block Logic: Check if the simulation should terminate.
            # Invariant: The loop terminates if no neighbors are returned by the supervisor.
            if self.neighbours is None:
                break

            # Block Logic: Wait until script assignments for the current timepoint are complete for the parent device.
            # Pre-condition: `self.device.timepoint_done` is an Event object.
            # Invariant: The worker threads proceed only after the supervisor signals completion of script assignment.
            self.device.timepoint_done.wait()

            # Block Logic: Process assigned scripts in a round-robin fashion.
            # Invariant: Each worker thread processes scripts where its `index` matches the script's distribution.
            for j in range(self.index, len(self.device.scripts), Device.nr_t):
                location = self.device.scripts[j][1] # Get location for the current script.
                script = self.device.scripts[j][0]   # Get script object.

                # Block Logic: Acquire a lock for the specific location to ensure exclusive access to its data.
                # Invariant: Only one worker thread can modify or read data for `location` at a time.
                for i in range(len(Device.location_locks)): # Iterate through shared location locks.
                    if location == Device.location_locks[i][0]: # Find the lock corresponding to the current location.
                        Device.location_locks[i][1].acquire()   # Acquire the lock.
                        break # Lock acquired, exit loop.

                script_data = [] # List to accumulate data for the current script's execution.
                
                # Block Logic: Collect data from neighboring devices for the current location.
                # Invariant: `script_data` will contain data from all available neighbors for the given location.
                for device in self.neighbours:
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
                    for device in self.neighbours:
                        device.set_data(location, result)
                    
                    # Functional Utility: Update the current device's own data with the script's result.
                    self.device.set_data(location, result)

                # Block Logic: Release the lock for the specific location.
                # Invariant: The lock is released after data access and modification are complete.
                for i in range(len(Device.location_locks)):
                    if location == Device.location_locks[i][0]:
                        Device.location_locks[i][1].release()
                        break # Lock released, exit loop.

            # Block Logic: Synchronize all `DeviceThread` instances across all devices using the global barrier.
            # Invariant: All DeviceThread instances in the simulation will reach this barrier before any proceeds.
            Device.barrier.wait()
            
            # Block Logic: Only the worker thread with index 0 performs cleanup for the next timepoint.
            # Invariant: `self.device.timepoint_done` is cleared.
            if self.index == 0:
                self.device.timepoint_done.clear()

            # Block Logic: Only the worker thread with index 0 clears the `neighbours_event` for the next timepoint.
            # Invariant: `self.device.neighbours_event` is cleared, resetting for the next neighbor fetch.
            if self.index == 0:
                self.device.neighbours_event.clear()
            Device.barrier.wait() # Another barrier wait, potentially for re-entering the main loop.


