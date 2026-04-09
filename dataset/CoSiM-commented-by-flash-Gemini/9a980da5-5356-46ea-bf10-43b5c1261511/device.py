"""
@file device.py
@brief Implements a simulated distributed device system for sensor data processing
       and communication. This module defines Device nodes, a custom ReusableBarrier
       for synchronization, and worker threads (MyThread) for concurrent script execution
       on local and neighboring device data.

Architectural Intent:
- Simulate a network of interconnected devices (e.g., sensor nodes).
- Support concurrent data processing via dedicated worker threads per script per timepoint.
- Coordinate execution across devices using a reusable barrier synchronization mechanism.
- Manage local sensor data and exchange data with neighboring devices with explicit locking per data location.

Domain: Distributed Systems, Concurrency, Simulation.
"""

from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    @brief Implements a reusable barrier synchronization mechanism using a double-phase approach.
           Threads wait at the barrier until all `num_threads` have arrived, allowing subsequent reuse.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier.
        @param num_threads The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Functional Utility: Counter for threads arriving at the first phase of the barrier.
        # Stored in a list to make it mutable and shareable between methods.
        self.count_threads1 = [self.num_threads]
        # Functional Utility: Counter for threads arriving at the second phase of the barrier.
        # Stored in a list to make it mutable and shareable between methods.
        self.count_threads2 = [self.num_threads]
        # Functional Utility: Lock to protect access to the thread counters.
        self.count_lock = Lock()
        # Functional Utility: Semaphore for synchronizing threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        # Functional Utility: Semaphore for synchronizing threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all threads have reached this point in both phases,
               then allows all to proceed.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Generic phase implementation for the reusable barrier.
        @param count_threads A mutable counter (list with one element) for the current phase.
        @param threads_sem The semaphore associated with the current phase.
        """
        # Block Logic: Atomically decrements the thread counter for the current phase.
        with self.count_lock:
            count_threads[0] -= 1
            # Block Logic: If this is the last thread to arrive at the current phase.
            # Invariant: `count_threads[0]` is zero, meaning all `num_threads` have arrived.
            if count_threads[0] == 0:
                # Functional Utility: Release all waiting threads from the semaphore.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Functional Utility: Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # Functional Utility: Blocks the thread until released by the last thread in this phase.
        threads_sem.acquire()

class Device(object):
    """
    @brief Represents a single device (node) in the simulated distributed system.
           Each device manages its own sensor data, communicates with a supervisor,
           and coordinates concurrent script execution.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id Unique identifier for this device.
        @param sensor_data Dictionary containing local sensor readings.
        @param supervisor Reference to the supervisor object for global coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Functional Utility: Event to signal when new scripts have been assigned.
        self.script_received = Event()
        # Functional Utility: List to hold tuples of (script, location) assigned to this device.
        self.scripts = []
        # Functional Utility: List to hold references to other devices in the system.
        self.devices = []
        # Functional Utility: Event to signal completion of processing for a specific timepoint.
        self.timepoint_done = Event()
        # Functional Utility: The main thread for this device, handling orchestration.
        self.thread = DeviceThread(self)
        # Functional Utility: Reference to the global barrier for synchronizing all Device instances.
        self.barrier = None
        # Functional Utility: List to keep track of worker threads (`MyThread`) created in each time step.
        self.list_thread = []
        self.thread.start()
        # Functional Utility: List of `Lock` objects, where `location_lock[idx]` protects data at `sensor_data[idx]`.
        # Initialized with `None` and filled dynamically. Assumes `location` is an integer index.
        self.location_lock = [None] * 100

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device with references to all other devices in the system.
               Initializes the global barrier if not already set, ensuring all devices share it.
        @param devices A list of all Device instances in the system.
        """
        # Block Logic: Initializes the global barrier instance if it hasn't been set yet.
        # This typically occurs once for the first device to call this method.
        # Invariant: After this block, all devices will share the same `ReusableBarrier` instance.
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            # Block Logic: Iterates through all devices to ensure they all reference the same barrier.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        # Block Logic: Populates the internal list of device neighbors.
        # Invariant: `self.devices` contains references to all other devices in the system.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by this device's worker threads at a specific data location.
               Also manages the creation or sharing of `Lock` objects for data locations.
        @param script The script object to be executed.
        @param location The data location (e.g., sensor ID) the script will operate on.
        """
        # Functional Utility: Flag to indicate if a lock for the location was found on another device.
        flag = 0

        # Block Logic: If a script is provided, adds it to the list of scripts and manages its associated lock.
        # Invariant: `self.scripts` contains pending (script, location) tuples.
        if script is not None:
            self.scripts.append((script, location))
            # Block Logic: Dynamically initializes a lock for the given `location` if it doesn't exist.
            # This logic also attempts to share an existing lock from another device if available.
            # Pre-condition: `location` is a valid index for `self.location_lock`.
            if self.location_lock[location] is None:
                # Block Logic: Searches other devices for an existing lock for this `location`.
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        flag = 1
                        break

                # Block Logic: If no existing lock was found, create a new one.
                if flag == 0:
                    self.location_lock[location] = Lock()
            # Functional Utility: Signals that a script has been received, potentially unblocking `DeviceThread`.
            self.script_received.set()
        else:
            # Functional Utility: Signals that processing for the current timepoint is done for this device.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location. Note: The lock for `location`
               must be acquired by the calling thread (e.g., `MyThread`) before calling this.
        @param location The location (key) for which to retrieve data.
        @return The sensor data or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location. Note: The lock for `location`
               must be held by the calling thread (e.g., `MyThread`) before calling this.
        @param location The location (key) for which to set data.
        @param data The data to be stored.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its main DeviceThread.
        """
        self.thread.join()


class MyThread(Thread):
    """
    @brief A worker thread responsible for executing a single script on data pertaining
           to a specific location, interacting with its parent device and neighbors.
           It handles locking for its assigned data location.
    """
    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes a MyThread instance.
        @param device The Device instance this thread belongs to.
        @param location The data location this thread will process.
        @param script The script to execute.
        @param neighbours A list of neighboring Device instances.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution loop for the MyThread.
               Acquires a lock, gathers data, executes the script, updates data, and releases the lock.
        """
        # Functional Utility: Acquires the lock for the specific data location, ensuring exclusive access.
        self.device.location_lock[self.location].acquire()
        # Functional Utility: List to aggregate data relevant to the current script.
        script_data = []
        
        # Block Logic: Gathers data for the current location from all neighboring devices.
        # Invariant: `script_data` contains available data from neighbors for the specified `location`.
        for device in self.neighbours:
            data = device.get_data(self.location) # Functional Utility: Retrieves data (assumes lock already held by caller or internal to get_data)
            if data is not None:
                script_data.append(data)
            
        # Functional Utility: Gathers data for the current location from the local device.
        data = self.device.get_data(self.location) # Functional Utility: Retrieves data (assumes lock already held by caller or internal to get_data)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script only if relevant data was collected.
        # Pre-condition: `script_data` is not empty.
        if script_data != []:
            # Functional Utility: Executes the assigned script with the collected data.
            result = self.script.run(script_data)
            
            # Block Logic: Updates the data at 'location' on all neighboring devices with the script's result.
            # Invariant: Neighboring devices' data is updated with consistency.
            for device in self.neighbours:
                device.set_data(self.location, result) # Functional Utility: Sets data (assumes lock already held by caller or internal to set_data)
                
            # Functional Utility: Updates the data at 'location' on the local device with the script's result.
            self.device.set_data(self.location, result) # Functional Utility: Sets data (assumes lock already held by caller or internal to set_data)
        # Functional Utility: Releases the lock for the specific data location.
        self.device.location_lock[self.location].release()


class DeviceThread(Thread):
    """
    @brief The main orchestration thread for a Device. Responsible for interacting with the supervisor,
           managing timepoints, and delegating script execution to individual worker threads (`MyThread`).
    """
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
               Handles fetching neighbors, creating and joining worker threads for script execution,
               and participating in global device synchronization.
        """
        # Functional Utility: Waits until all devices have completed their initial setup (handled by Device.setup_devices).
        # Note: The `Event` signaling global setup is implicit through `self.device.barrier` in this design.
        
        # Block Logic: Main loop for processing timepoints and supervisor interactions.
        while True:
            # Functional Utility: Fetches information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: If no neighbors are returned (signaling simulation end), break the loop.
            # Invariant: `neighbours` is None when the simulation is to terminate.
            if neighbours is None:
                break

            # Functional Utility: Waits for the supervisor to signal timepoint completion for this device.
            self.device.timepoint_done.wait()

            # Functional Utility: Clears previous worker threads from the list.
            self.device.list_thread = []

            # Block Logic: Creates and starts a `MyThread` for each assigned script.
            # Invariant: Each script is assigned to its own worker thread for concurrent execution.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            # Block Logic: Starts all worker threads and then waits for their completion.
            # Invariant: All scripts assigned for the current timepoint are executed.
            for thread_elem in self.device.list_thread:
                thread_elem.start()
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            # Functional Utility: Clears the list of worker threads for the next timepoint.
            self.device.list_thread = []

            # Functional Utility: Clears the timepoint_done event for the next cycle.
            self.device.timepoint_done.clear()
            # Functional Utility: Global synchronization point: waits for all devices to complete their timepoint processing.
            self.device.barrier.wait()
