"""
@file device.py
@brief Implements a simulated distributed device system for processing sensor data and managing
       inter-device communication. This module defines a multi-threaded architecture for each device,
       including a custom reusable barrier, a main device thread, and worker core threads for concurrent
       script execution with fine-grained data locking.

Architectural Intent:
- Simulate a network of interconnected devices (e.g., sensor nodes).
- Support concurrent data processing via multiple worker threads (`DeviceCore`) within each device.
- Coordinate execution across all devices (`Device` instances) using a global reusable barrier.
- Manage local sensor data and facilitate data exchange with neighboring devices, ensuring thread safety
  through explicit locking on shared data locations.
- Dynamically assign and execute processing scripts per timepoint.

Domain: Distributed Systems, Concurrency, Multi-threading, Simulation, Sensor Networks.
"""

from threading import Event, Thread, Condition, Lock
from Queue import Queue

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier synchronization mechanism.
           Threads wait at the barrier until all `num_threads` have arrived, allowing subsequent reuse.
           This implementation uses a `Condition` object for managing waiting and notification.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier.
        @param num_threads The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Functional Utility: Counter for threads currently waiting at the barrier.
        self.count_threads = self.num_threads
        # Functional Utility: Condition variable for managing thread waiting and notification.
        self.cond = Condition()

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have called `wait()`.
               Once all threads arrive, they are all released simultaneously.
               The barrier then resets for future use.
        """
        # Block Logic: Ensures exclusive access to the counter and condition variable.
        self.cond.acquire()
        self.count_threads -= 1
        # Block Logic: Checks if this is the last thread to arrive at the barrier.
        # Invariant: `self.count_threads` is zero, meaning all `num_threads` have arrived.
        if self.count_threads == 0:
            self.cond.notify_all() # Functional Utility: Notifies all waiting threads to proceed.
            self.count_threads = self.num_threads # Functional Utility: Resets the counter for barrier reuse.
        else:
            self.cond.wait() # Functional Utility: Releases the lock and waits until notified by the last thread.
        self.cond.release() # Functional Utility: Releases the lock after proceeding.

class Device(object):
    """
    @brief Represents a single device (node) in the simulated distributed system.
           Each device manages its own sensor data, communicates with a supervisor,
           and orchestrates concurrent script execution across its worker cores.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id Unique identifier for this device.
        @param sensor_data Dictionary containing local sensor readings keyed by location.
        @param supervisor Reference to the supervisor object for global coordination and neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Functional Utility: Reference to the global barrier for synchronizing all Device instances.
        self.barrier = None
        # Functional Utility: Reference to a global lock protecting the shared `locations` list.
        self.locations_mutex = None
        # Functional Utility: Event signaling that global device setup is complete and this device's thread can begin.
        self.can_begin = Event()
        # Functional Utility: Event to signal that all location locks have been computed/initialized (currently unused).
        self.locks_computed = Event()
        # Functional Utility: Event signaling that the processing for the current timepoint is complete for this device.
        self.timepoint_done = Event()
        # Functional Utility: Event signaling that the overall simulation should terminate.
        self.simulation_end = Event()
        # Functional Utility: Local lock for this specific device, used to protect its `sensor_data` when accessed by `DeviceCore`s.
        self.lock = Lock()
        # Functional Utility: Queue to hold scripts assigned to this device for processing in the current timepoint.
        self.scripts_queue = Queue()
        # Functional Utility: List to store all scripts assigned to this device over its lifetime.
        self.scripts = []
        # Functional Utility: Global list of all unique data locations across all devices, protected by `locations_mutex`.
        self.locations = []
        # Functional Utility: A list of `Lock` objects, where `locations_locks[idx]` protects data at `locations[idx]`.
        # This list is dynamically populated by device 0 and shared amongst all devices.
        self.locations_locks = [] # This is added here based on DeviceThread.run logic.
        # Functional Utility: List to hold references to other Device instances in the system.
        self.devices = [] # This is added here based on setup_devices logic.
        # Functional Utility: The main thread for this device, handling orchestration.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device with references to all other devices in the system.
               Performs global initialization for shared resources (barrier, locations_mutex)
               if this is device 0, and propagates them to all other devices.
        @param devices A list of all Device instances in the system.
        """
        # Block Logic: Device 0 performs global initialization for shared resources.
        # Invariant: `self.barrier` and `self.locations_mutex` are initialized once and shared globally.
        if self.device_id == 0:
            # Functional Utility: Initializes the global reusable barrier with the total number of devices.
            self.barrier = ReusableBarrier(len(devices))
            # Functional Utility: Stores references to all devices.
            self.devices = devices
            # Functional Utility: Initializes the global mutex for protecting the shared `locations` list.
            self.locations_mutex = Lock()

            # Block Logic: Propagates the initialized shared resources to all other devices.
            # Invariant: All devices reference the same global barrier and locations_mutex.
            for device in devices:
                device.locations_mutex = self.locations_mutex
                device.locations = self.locations
                device.barrier = self.barrier
                device.devices = devices # Ensure all devices know about each other
                device.can_begin.set() # Functional Utility: Allows DeviceThread to start execution.

            self.can_begin.set() # Functional Utility: Allows DeviceThread of device 0 to start.
        else:
            # Block Logic: Non-zero devices wait for Device 0 to complete global setup.
            # Invariant: This device's `can_begin` event is set by Device 0.
            pass # The self.can_begin.wait() in DeviceThread.run handles this synchronization

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by this device's worker cores at a specific data location.
               Scripts are added to an internal queue for processing.
        @param script The script object to be executed. If None, it signals the end of scripts for a timepoint.
        @param location The data location (e.g., sensor ID) the script will operate on.
        """
        # Block Logic: If a valid script is provided, it's added to the scripts list and queue.
        # Invariant: `self.scripts_queue` contains (script, location) pairs to be processed.
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_queue.put((script, location))
        else:
            # Functional Utility: Signals that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
               Note: This method assumes the caller (e.g., `DeviceCore` or `MyThread`)
               manages the necessary locks for thread safety.
        @param location The location (key) for which to retrieve data.
        @return The sensor data value if found, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
               Note: This method assumes the caller (e.g., `DeviceCore` or `MyThread`)
               manages the necessary locks for thread safety.
        @param location The location (key) for which to set data.
        @param data The data to be stored.
        """
        # Block Logic: Updates `sensor_data` only if the location key already exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its main DeviceThread.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief The main orchestration thread for a Device. Responsible for interacting with the supervisor,
           managing timepoints, dynamically discovering locations, and delegating script execution
           to its pool of `DeviceCore` worker threads.
    """
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Functional Utility: Stores a reference to the current list of neighboring devices received from the supervisor.
        self.current_neighbours = []
        # Functional Utility: Creates a pool of `DeviceCore` worker threads for concurrent script execution.
        self.cores = [DeviceCore(self, i, self.device.simulation_end) for i in xrange(0, 8)] # Assuming 8 cores per device.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
               Handles initial setup, location discovery, `DeviceCore` management,
               and timepoint-based processing and synchronization.
        """
        # Functional Utility: Waits until Device 0 has completed its global setup and signaled `can_begin`.
        self.device.can_begin.wait()

        # Block Logic: Acquires the global mutex to safely update the shared `locations` list.
        # Invariant: `self.device.locations` eventually contains all unique data locations across all devices.
        self.device.locations_mutex.acquire()
        # Block Logic: Adds this device's unique sensor data locations to the global shared `locations` list.
        for location in self.device.sensor_data.keys():
            if location not in self.device.locations:
                self.device.locations.append(location)
        self.device.locations_mutex.release() # Functional Utility: Releases the global mutex.

        # Functional Utility: Synchronizes all DeviceThreads, ensuring `locations` list is fully populated.
        self.device.barrier.wait()

        # Block Logic: Device 0 initializes the shared `locations_locks` list after all locations are known.
        # Invariant: `self.device.locations_locks` contains a `Lock` for each unique location, shared across devices.
        if self.device.device_id == 0:
            self.device.locations_locks = [Lock() for _ in xrange(0, len(self.device.locations))]
            # Block Logic: Propagates the initialized `locations_locks` list to all other devices.
            for device in self.device.devices:
                device.locations_locks = self.device.locations_locks

        # Functional Utility: Synchronizes all DeviceThreads again after `locations_locks` initialization.
        self.device.barrier.wait()

        # Functional Utility: Starts all `DeviceCore` worker threads.
        for core in self.cores:
            core.start()

        # Block Logic: Main loop for processing timepoints and managing `DeviceCore`s.
        # Invariant: Each timepoint involves fetching neighbors, assigning scripts, and awaiting core completion.
        while True:
            # Block Logic: Empties the `scripts_queue` of any remaining scripts from previous timepoints.
            # Invariant: `scripts_queue` is empty at the start of processing new timepoint scripts.
            while not self.device.scripts_queue.empty():
                self.device.scripts_queue.get()

            # Block Logic: Re-populates the `scripts_queue` with current timepoint scripts.
            # Invariant: `scripts_queue` now contains all scripts assigned for the current timepoint.
            for script in self.device.scripts:
                self.device.scripts_queue.put(script)

            # Functional Utility: Fetches information about neighboring devices from the supervisor for the current timepoint.
            self.current_neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Checks if the supervisor has signaled the end of the simulation.
            # Invariant: `self.current_neighbours` is None when the simulation should terminate.
            if self.current_neighbours is None:
                self.device.simulation_end.set() # Functional Utility: Signals all `DeviceCore`s to terminate.
                # Functional Utility: Signals `DeviceCore`s that new scripts are available (even if it's termination signal).
                for core in self.cores:
                    core.got_script.set()
                # Block Logic: Waits for all `DeviceCore` worker threads to finish and join.
                for core in self.cores:
                    core.join()
                break # Break from the main `while True` loop, ending the `DeviceThread`.

            # Block Logic: Distributes scripts from `scripts_queue` to available `DeviceCore`s for execution.
            # Invariant: All scripts in `scripts_queue` are eventually assigned to a `DeviceCore` and executed.
            while not self.device.timepoint_done.isSet() or not self.device.scripts_queue.empty():
                if not self.device.scripts_queue.empty():
                    script, location = self.device.scripts_queue.get()

                    # Block Logic: Finds an available `DeviceCore` to assign the script to.
                    # Invariant: An available core is found to process the script.
                    core_found = False
                    while not core_found:
                        for core in self.cores:
                            if core.running is False:
                                core_found = True
                                core.script = script
                                core.location = location
                                core.neighbours = self.current_neighbours
                                core.running = True
                                core.got_script.set() # Functional Utility: Signals the core to start processing the script.
                                break

            # Functional Utility: Global synchronization point: waits for all `DeviceThread`s to complete script distribution.
            self.device.barrier.wait()

            # Functional Utility: Clears the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()

class DeviceCore(Thread):
    """
    @brief A worker thread that operates within a `DeviceThread`. It processes assigned scripts
           on specific data locations, gathering data from its parent device and neighbors,
           executing the script, and updating results, ensuring thread safety with explicit locks.
    """
    def __init__(self, device_thread, core_id, simulation_end):
        """
        @brief Initializes a DeviceCore instance.
        @param device_thread The parent `DeviceThread` that manages this core.
        @param core_id Unique identifier for this core.
        @param simulation_end An `Event` to signal when the overall simulation should terminate.
        """
        Thread.__init__(self, name="Device Core %d" % core_id)
        self.device_thread = device_thread
        self.core_id = core_id
        # Functional Utility: List of neighboring devices for script data gathering.
        self.neighbours = []
        # Functional Utility: Event signaling that a new script has been assigned to this core.
        self.got_script = Event()
        # Functional Utility: Flag indicating if this core is currently running a script.
        self.running = False
        self.simulation_end = simulation_end
        # Functional Utility: Placeholder for the script to be executed.
        self.script = None
        # Functional Utility: Placeholder for the data location the script will operate on.
        self.location = None

    def run(self):
        """
        @brief The main execution loop for the DeviceCore.
               Waits for script assignments, executes them, handles data synchronization,
               and terminates when the simulation ends.
        """
        # Block Logic: Continuously processes scripts until the simulation ends.
        while True:
            # Functional Utility: Waits until the `DeviceThread` assigns a new script (sets `got_script`).
            self.got_script.wait()

            # Block Logic: Checks if the simulation has been signaled to end while waiting for a script.
            if self.simulation_end.isSet():
                break # Break from the `while True` loop, terminating the `DeviceCore`.

            # Functional Utility: Acquires the specific lock for the assigned data location, ensuring exclusive access.
            # Pre-condition: `self.location` is a valid index for `self.device_thread.device.locations_locks`.
            self.device_thread.device.locations_locks[self.location].acquire()

            # Functional Utility: List to aggregate data relevant to the current script's execution.
            script_data = []
            # Block Logic: Gathers data for the specified location from all neighboring devices.
            # Invariant: `script_data` contains available data from neighbors for the given `location`.
            for neighbour in self.neighbours:
                # Functional Utility: Acquires local lock of the neighbor device to safely retrieve data.
                neighbour.lock.acquire()
                data = neighbour.get_data(self.location)
                neighbour.lock.release() # Functional Utility: Releases the neighbor device's local lock.
                if data is not None:
                    script_data.append(data)

            # Functional Utility: Acquires local lock of the parent device to safely retrieve its data.
            self.device_thread.device.lock.acquire()
            data = self.device_thread.device.get_data(self.location)
            self.device_thread.device.lock.release() # Functional Utility: Releases the parent device's local lock.
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script only if relevant data was collected.
            if script_data != []:
                # Functional Utility: Executes the assigned script with the collected data.
                result = self.script.run(script_data)

                # Functional Utility: Acquires local lock of the parent device to safely update its data.
                self.device_thread.device.lock.acquire()
                self.device_thread.device.set_data(self.location, result)
                self.device_thread.device.lock.release() # Functional Utility: Releases the parent device's local lock.

                # Block Logic: Updates the data at 'location' on all neighboring devices with the script's result.
                # Invariant: Neighboring devices' data is updated with consistency.
                for neighbour in self.neighbours:
                    # Functional Utility: Acquires local lock of the neighbor device to safely update data.
                    neighbour.lock.acquire()
                    neighbour.set_data(self.location, result)
                    neighbour.lock.release() # Functional Utility: Releases the neighbor device's local lock.

            # Functional Utility: Releases the specific lock for the assigned data location.
            self.device_thread.device.locations_locks[self.location].release()

            # Functional Utility: Resets the `running` flag to indicate availability for new scripts.
            self.running = False

            # Functional Utility: Clears the `got_script` event, preparing to wait for the next script assignment.
            self.got_script.clear()

            # Block Logic: Final check for simulation termination after processing a script.
            if self.simulation_end.isSet():
                break # Break from the `while True` loop, terminating the `DeviceCore`.
