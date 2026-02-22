

"""
@brief This module defines `Device`, `DeviceThread`, `DeviceWorker` classes, and a custom `ReusableBarrier`
for simulating a distributed system.
@details It features a semaphore-based reusable barrier for inter-device synchronization, employs a main
device thread that manages and distributes scripts to a pool of worker threads, and uses locks
for thread-safe data access and coordination within the simulation.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue

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
            # Block Logic: If this is the last thread in this phase, release all waiting threads.
            # Invariant: All `num_threads` threads are unblocked.
            if count_threads[0] == 0:
                for i in range(self.num_threads): # Release all `num_threads` waiting threads.
                    threads_sem.release()
                # Functional Utility: Reset the counter for this phase for reusability.
                count_threads[0] = self.num_threads
        threads_sem.acquire() # Acquire the semaphore, waiting if not yet released by the last thread.


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It utilizes a shared `ReusableBarrier` (`neighbours_barrier`) for global synchronization among devices
    and a `Lock` (`set_lock`) for protecting its sensor data during updates. `DeviceThread` manages
    script execution and worker threads.
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, implementing explicit synchronization and mutual exclusion
    for coordinated and consistent execution across timepoints.
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
        self.result_queue = Queue.Queue() # Queue to store results from worker threads (not explicitly used in provided code).
        self.set_lock = Lock()            # Lock for protecting `self.sensor_data` during updates in `set_data`.
        self.neighbours_lock = None       # Reference to a shared Lock for protecting access to supervisor's neighbours.
        self.neighbours_barrier = None    # Reference to a shared ReusableBarrier for inter-device synchronization.

        self.script_received = Event() # Event to signal when new scripts are ready for execution.
        self.scripts = []            # List to store assigned scripts and their locations.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's script assignment.

        self.thread = DeviceThread(self) # The main worker thread for this device.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives (barrier and a global lock) for all devices.
        @details This method ensures that a single `ReusableBarrier` instance (`neighbours_barrier`)
        and a single `Lock` instance (`neighbours_lock`) are created only once by device 0
        and then distributed to all other devices in the simulation. This centralizes the management
        of these shared resources. Finally, it starts the device's main `DeviceThread`.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of shared synchronization
                     and mutual exclusion primitives, and starting the main device thread.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `self.neighbours_barrier` refers to a globally shared `ReusableBarrier` instance after setup.
                   `self.neighbours_lock` refers to a globally shared `Lock` instance after setup.
                   The device's `DeviceThread` is started.
        """
        # Block Logic: Initialize shared lock and barrier only once by the device identified as the "first" device.
        # This implementation assumes device 0 is the first device in the list for initialization.
        # Invariant: Global `neighbours_lock` and `neighbours_barrier` become initialized and shared across devices.
        if self.device_id == devices[0].device_id: # Only device 0 performs initialization.
            self.neighbours_lock = Lock() # Create a shared lock for protecting neighbor access.
            self.neighbours_barrier = ReusableBarrier(len(devices)) # Create a reusable barrier, sized for all devices.
        else:
            # Other devices reference the shared lock and barrier created by device 0.
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier

        self.thread.start() # Start the device's main execution thread.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location.
        @details If a script is provided, it's appended to the device's script queue.
        The `script_received` event is always set (regardless of script content) to signal
        the `DeviceThread` that scripts are available for processing. If no script is provided
        (i.e., `script` is None), it also sets the `timepoint_done` event to unblock the `DeviceThread`.
        @param script (object): The script object to be executed, or None to signal end of assignments.
        @param location (str): The location associated with the script or data.
        @block_logic Handles script assignment and signals readiness for execution and timepoint completion.
        @pre_condition `self.scripts` is a list, `self.script_received` and `self.timepoint_done` are Event objects.
        @invariant Either a script is added to `self.scripts` or `timepoint_done` is set, and `script_received` is always set.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the queue.
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.script_received.set() # Signal that current batch of scripts has been received (even if empty).
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
        @brief Sets or updates sensor data for a specific location in a thread-safe manner.
        @details This method acquires `set_lock` before modifying `self.sensor_data` to ensure
        thread safety during updates.
        @param location (str): The location whose data is to be updated.
        @param data (object): The new data value for the specified location.
        @block_logic Provides thread-safe access to update internal sensor data.
        @pre_condition `self.sensor_data` is a dictionary, `self.set_lock` is an initialized Lock.
        @invariant If `location` is a key in `self.sensor_data`, its value is updated under lock protection.
        """
        self.set_lock.acquire() # Acquire lock to ensure thread-safe modification of sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_lock.release() # Release lock after update.

    def shutdown(self):
        """
        @brief Shuts down the device by joining its associated thread.
        @details This ensures that the device's worker thread completes its execution before the program exits.
        """
        self.thread.join()



class DeviceThread(Thread):
    """
    @brief The main controlling thread for a Device instance.
    @details This thread orchestrates the device's operational cycle for each timepoint.
    It manages the fetching of neighbor information, waits for script assignments, and
    distributes these scripts among a pool of `DeviceWorker` threads for concurrent execution.
    It coordinates with the global `ReusableBarrier` and manages `neighbours_lock` for thread safety.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    abstracting multi-threaded script execution through worker threads and ensuring proper
    coordination and data consistency within the distributed system.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
        self.device = device # Reference to the parent Device object.
        self.workers = [] # List to hold DeviceWorker instances.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        @details This method continuously monitors the simulation state. For each timepoint,
        it acquires `self.device.neighbours_lock` to fetch neighbor information from the supervisor.
        If neighbors are available, it waits until `script_received` is set (signaling that script
        assignments are complete), then creates a pool of 8 `DeviceWorker` threads. It distributes
        the assigned scripts among these workers, starts them, and waits for their completion.
        Finally, it clears `script_received`, resets the script list, and synchronizes with other
        `DeviceThread` instances via the global `ReusableBarrier` (`neighbours_barrier`).
        The loop terminates when the supervisor signals the end of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        parallel script execution, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `scripts` list, `script_received` event, `neighbours_lock`, `neighbours_barrier`.
        @invariant The thread progresses through timepoints, processes scripts concurrently,
                   and ensures global synchronization.
        """
        while True:
            # Block Logic: Acquire lock to safely fetch neighbor information from the supervisor.
            # Invariant: `neighbours` list is obtained under mutual exclusion.
            self.device.neighbours_lock.acquire()
            neighbours = self.device.supervisor.get_neighbours() # Functional Utility: Get neighbours.
            self.device.neighbours_lock.release() # Release lock.

            # Block Logic: Check if the simulation should terminate.
            # Pre-condition: `neighbours` list indicates the current state of the network.
            # Invariant: The loop terminates if no neighbors are returned by the supervisor.
            if neighbours is None:
                break

            # Block Logic: Wait until script assignments for the current timepoint are complete.
            # Pre-condition: `self.device.script_received` is an Event object.
            # Invariant: The thread proceeds only after the supervisor signals completion of script assignment.
            self.device.script_received.wait()

            # Functional Utility: Clear the list of workers for the new timepoint.
            self.workers = []
            # Block Logic: Create a fixed pool of 8 `DeviceWorker` threads.
            # Invariant: `self.workers` contains 8 new `DeviceWorker` instances.
            for i in range(8):
                self.workers.append(DeviceWorker(self.device, i, neighbours))

            # Block Logic: Distribute assigned scripts among the worker threads.
            # Invariant: Each script from `self.device.scripts` is added to a `DeviceWorker`.
            for (script, location) in self.device.scripts:
                # Functional Utility: Find a worker for this script/location, using a simple load balancing (not explicitly shown).
                # The provided code seems to assign based on location, but the 'added' logic is flawed.
                # It appears the intent is to distribute scripts to workers, but the logic is missing a clear distribution strategy.
                # Assuming the goal is to load-balance scripts or ensure workers know their scripts.
                added = False
                for worker in self.workers:
                    if location in worker.locations: # Checks if the worker already has this location.
                        worker.add_script(script, location) # Add script to existing worker.
                        added = True
                        break # Script added, move to next script.

                if added == False: # If no worker explicitly 'had' the location, assign to the worker with least load.
                    minimum = len(self.workers[0].locations) # Start with first worker's load.
                    chosen_worker = self.workers[0]
                    for worker in self.workers: # Find worker with minimum assigned locations.
                        if minimum > len(worker.locations):
                            minimum = len(worker.locations)
                            chosen_worker = worker

                    chosen_worker.add_script(script, location) # Assign script to chosen worker.

            # Block Logic: Start all `DeviceWorker` instances concurrently.
            # Invariant: All `DeviceWorker` instances begin their `run` method concurrently.
            for worker in self.workers:
                worker.start()

            # Block Logic: Wait for all `DeviceWorker` instances to complete their execution.
            # Invariant: The DeviceThread will not proceed until all its `DeviceWorker` children have finished.
            for worker in self.workers:
                worker.join()

            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances in the simulation will reach this barrier before any proceeds to the next timepoint.
            self.device.neighbours_barrier.wait()
            # Functional Utility: Clear the `script_received` event and scripts list for the next timepoint.
            self.device.script_received.clear()
            self.device.scripts = [] # Reset scripts list for the next timepoint.


class DeviceWorker(Thread):
    """
    @brief A worker thread dedicated to executing a subset of assigned scripts for a Device instance.
    @details This thread receives a list of scripts and their associated locations, collects data from
    neighbors and the parent device, executes the scripts' logic, and updates sensor data.
    @architectural_intent Enhances parallelism by distributing script execution among a fixed pool
    of threads within a `Device`, thereby improving throughput.
    """
    
    def __init__(self, device, worker_id, neighbours):
        """
        @brief Initializes a new DeviceWorker instance.
        @param device (Device): The parent Device object that this worker thread serves.
        @param worker_id (int): A unique identifier for this specific worker thread within its parent Device.
        @param neighbours (list): A list of neighboring Device objects from which to collect sensor data.
        """
        Thread.__init__(self) # Initialize the base Thread class.
        self.device = device # Reference to the parent Device object.
        self.worker_id = worker_id # Unique ID of this worker thread.
        self.scripts = [] # List to hold scripts assigned to this worker.
        self.locations = [] # List to hold locations corresponding to scripts assigned to this worker.
        self.neighbours = neighbours # List of neighboring devices.

    def add_script(self, script, location):
        """
        @brief Adds a script and its location to this worker's processing queue.
        @param script (object): The script object to be executed.
        @param location (str): The location associated with the script.
        @block_logic Appends the script and location to internal lists for later execution.
        @pre_condition `script` and `location` are valid.
        @invariant `self.scripts` and `self.locations` are updated with the new entry.
        """
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        """
        @brief Executes all scripts assigned to this worker.
        @details This method iterates through its assigned scripts. For each script, it collects data
        from neighbors and the parent device, executes the script, and updates sensor data.
        The data update operations are protected by `self.device.set_lock`.
        @block_logic Processes a batch of scripts, performing data collection, execution, and update.
        @pre_condition `self.scripts` and `self.locations` are aligned lists of scripts and locations.
                       `self.device.set_lock` is available for data updates.
        @invariant Each script is executed, and data is updated on the device and its neighbors.
        """
        # Block Logic: Iterate through all assigned scripts and their corresponding locations.
        # Invariant: Each script is processed from data collection to data update.
        for (script, location) in zip(self.scripts, self.locations):
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
            # Invariant: `res` holds the output of the script's execution.
            if script_data != []:
                res = script.run(script_data) # Execute the script with the collected data.

                # Block Logic: Propagate the script's result to neighboring devices and own device.
                # Invariant: All relevant devices receive the updated data for the given location.
                for device in self.neighbours:
                    device.set_data(location, res) # Updates neighbor's data (protected by neighbour's `set_lock`).
                self.device.set_data(location, res) # Updates own device's data (protected by own `set_lock`).

    def run(self):
        """
        @brief The main execution loop for the DeviceWorker thread.
        @details This method simply calls `run_scripts` to execute all scripts assigned to it.
        @block_logic Executes assigned scripts.
        @pre_condition Scripts and locations are assigned to this worker.
        @invariant All scripts assigned to this worker are executed.
        """
        self.run_scripts()

