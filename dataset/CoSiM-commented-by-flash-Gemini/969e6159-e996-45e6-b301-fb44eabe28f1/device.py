"""
@969e6159-e996-45e6-b301-fb44eabe28f1/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices with a custom reentrant semaphore barrier.

This module defines the core components for simulating a network of sensor devices.
It features a `ReusableBarrierSem` implemented directly within the module for
efficient synchronization. Each `Device` operates with multiple worker threads
(`SingleDeviceThread`) that execute scripts, manage local sensor data, and
interact with neighbors. Synchronization is managed both within a device and
across devices using these barriers and location-specific locks.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate under the guidance of a supervisor.

Classes:
- Device: Represents a single simulated sensor device, orchestrating its workers and synchronization.
- DeviceThread: The main thread for a device, managing timepoint progression and worker threads.
- SingleDeviceThread: A worker thread responsible for executing individual scripts for specific locations.
- ReusableBarrierSem: A custom reentrant barrier implementation using semaphores.

Domain: Concurrent Programming, Distributed Systems Simulation, Parallel Processing, Custom Synchronization Primitives.
"""

from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each `Device` manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. It coordinates
    its worker threads and synchronizes with other devices using a global barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data, keyed by location.
        @param supervisor: A reference to the central supervisor managing all devices.
        """
        # Unique identifier for this device.
        self.device_id = device_id
        # Dictionary storing sensor data, keyed by location.
        self.sensor_data = sensor_data
        # Reference to the central supervisor.
        self.supervisor = supervisor
        # Event to signal that a script has been received for the current timepoint.
        self.script_received = Event() # This event appears unused in the provided code.
        # List to store assigned scripts, each being a tuple of (script, location).
        self.scripts = []
        # Event to signal that all scripts for the current timepoint have been assigned.
        self.timepoint_done = Event()
        # The main thread responsible for the device's lifecycle.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up global synchronization resources for all devices.

        This method designates the device with the smallest `device_id` as the
        "master" to initialize the global `ReusableBarrierSem` and a dictionary
        of `Lock` objects (`map_locations`) for location-specific data protection.
        Other devices (`dev`) receive references to these shared resources from the master.

        @param devices: A list of all Device instances in the simulation.
        """
        flag = True # Flag to determine if this device should be the master.
        device_number = len(devices)

        # Block Logic: Determines if the current device has the smallest ID among active devices.
        # If true, it assumes the role of the master for setting up shared resources.
        for dev in devices:
            if self.device_id > dev.device_id:
                flag = False # Another device has a smaller ID, so this is not the master.

        if flag == True:
            # Block Logic: Master device initialization of global resources.
            # Pre-condition: This device is determined to be the master.
            # Invariant: A global `ReusableBarrierSem` and `map_locations` (dictionary of locks)
            # are initialized and distributed to all participating devices.
            barrier = ReusableBarrierSem(device_number) # Global inter-device barrier.
            map_locations = {} # Dictionary to store Locks for each unique location.
            tmp = {} # Temporary dictionary (unused in the current logic).
            
            for dev in devices:
                dev.barrier = barrier # Distribute the global barrier.
                # Inline: Identifies new locations from current device's sensor_data not yet in `map_locations`.
                tmp = list(set(dev.sensor_data) - set(map_locations))
                for i in tmp:
                    map_locations[i] = Lock() # Create a new Lock for each new unique location.
                dev.map_locations = map_locations # Distribute the shared map of location locks.
                tmp = {} # Reset temporary dictionary.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed for a specific location on this device.

        If `script` is None, it signals that all scripts for the current timepoint
        have been assigned, and sets the `timepoint_done` event.

        @param script: The script object to be executed, or None to signal completion.
        @param location: The location pertinent to the script execution.
        """
        if script is not None:
            # Pre-condition: `script` is not None.
            # Invariant: The script is added to the device's list of pending scripts.
            self.scripts.append((script, location))
            self.script_received.set() # Signals that a script has been received.
        else:
            # Pre-condition: `script` is None, signaling end of script assignment for this timepoint.
            # Invariant: The `timepoint_done` event is set, signaling workers to begin processing.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The location for which to retrieve data.
        @return: The sensor data for the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The location for which to set data.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's main thread.

        Waits for the device's main thread to complete its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main execution thread for a `Device` instance.

    This thread manages timepoint progression, coordinates with the supervisor,
    and spawns `SingleDeviceThread` workers to execute scripts. It also handles
    inter-device synchronization using a global barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        @param device: The `Device` instance that this thread will manage.
        """
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Pre-condition: The device and its synchronization mechanisms are properly set up.
        Invariant: The device continuously processes timepoints, spawns workers to execute
                   assigned scripts, and synchronizes with other devices.
        """
        while True:
            # Inline: Clears the `timepoint_done` event for the current timepoint.
            self.device.timepoint_done.clear()
            # Block Logic: Fetches the current neighbors of this device from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Pre-condition: `neighbours` is None, indicating a termination signal from the supervisor.
                # Invariant: The loop breaks, leading to thread termination.
                break
            # Block Logic: Waits for the supervisor to signal that all scripts for the current
            # timepoint have been assigned and are ready for processing.
            self.device.timepoint_done.wait()
            
            script_list = [] # Local list to hold scripts.
            thread_list = [] # List to hold references to spawned worker threads.
            index = 0 # Starting index for distributing scripts (not clearly used for distribution here).

            # Block Logic: Copies scripts from the device's script list to a local list.
            for script in self.device.scripts:
                script_list.append(script)
            
            # Block Logic: Spawns 8 `SingleDeviceThread` workers.
            # This implementation seems to spawn 8 threads regardless of the number of scripts,
            # and each thread attempts to pop from `script_list` at a fixed `index`. This
            # design is potentially problematic for shared list manipulation without external synchronization.
            for i in xrange(8):
                thread = SingleDeviceThread(self.device, script_list, neighbours, index)
                thread.start()
                thread_list.append(thread)
            
            # Block Logic: Waits for all spawned worker threads to complete their execution.
            for i in xrange(len(thread_list)):
                thread_list[i].join()
            
            # Block Logic: Synchronizes with all other DeviceThreads using the global barrier.
            # This ensures all devices have finished processing their scripts before proceeding.
            self.device.barrier.wait()


class SingleDeviceThread(Thread):
    """
    @brief A worker thread responsible for executing individual scripts for specific locations.

    Each `SingleDeviceThread` processes a script, collects data from neighbors,
    executes the script, and updates relevant data in a thread-safe manner
    using location-specific locks.
    """
    
    def __init__(self, device, script_list, neighbours, index):
        """
        @brief Initializes a SingleDeviceThread worker.

        @param device: The parent `Device` instance.
        @param script_list: A list of scripts to be processed (shared among workers).
        @param neighbours: A list of neighboring `Device` instances to interact with.
        @param index: An index used to select a script from `script_list`.
        """
        Thread.__init__(self)
        self.device = device
        self.script_list = script_list
        self.neighbours = neighbours
        self.index = index

    def run(self):
        """
        @brief The main execution method for the SingleDeviceThread.

        Pre-condition: `script_list` is populated, and device/neighbor references are valid.
        Invariant: The thread attempts to execute a script, collecting and updating data,
                   while ensuring thread-safe access to location-specific resources.
        """
        # Block Logic: Checks if there are scripts to process.
        if self.script_list != []:
            # Inline: Attempts to pop a script from `script_list` at `self.index`.
            # This design may lead to `IndexError` if `self.index` is out of bounds or
            # if multiple threads try to pop from the same index concurrently without proper synchronization.
            # A more robust solution for shared list access would involve explicit locking.
            (script, location) = self.script_list.pop(self.index)
            self.compute(script, location)

    def update(self, result, location):
        """
        @brief Updates the sensor data on neighboring devices and the current device.

        @param result: The result of the script execution.
        @param location: The location whose data needs to be updated.
        """
        # Block Logic: Iterates through neighboring devices and updates their data.
        for device in self.neighbours:
            device.set_data(location, result)
        # Updates the current device's data.
        self.device.set_data(location, result)

    def collect(self, location, neighbours, script_data):
        """
        @brief Collects sensor data for a given location from neighboring devices and the current device.

        @param location: The location for which to collect data.
        @param neighbours: A list of neighboring `Device` instances.
        @param script_data: A list to append the collected data.
        """
        # Critical Section: Acquire the location-specific lock to ensure exclusive access
        # to data for this `location` during data collection.
        self.device.map_locations[location].acquire()
        # Block Logic: Gathers data from neighboring devices for the current `location`.
        for device in self.neighbours:
            data = device.get_data(location)
            if data is None:
                pass # If no data, do nothing.
            else:
                script_data.append(data)

        # Block Logic: Gathers data from its own sensor_data for the current `location`.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

    def compute(self, script, location):
        """
        @brief Executes the given script after collecting relevant data and updates results.

        @param script: The script object to execute.
        @param location: The location pertinent to the script execution.
        """
        script_data = [] # List to store data collected for the script.
        # Collects data for the script.
        self.collect(location, self.neighbours, script_data)

        if script_data == []:
            pass # If no data to process, do nothing.
        else:
            # Executes the script with the collected data.
            result = script.run(script_data)
            # Updates the data on neighbors and current device with the script's result.
            self.update(result, location)

        # Releases the location-specific lock after computation and updates are done.
        self.device.map_locations[location].release()

class ReusableBarrierSem():
    """
    @brief A custom reentrant barrier implementation using Semaphores (a two-phase barrier).

    This barrier allows multiple threads to wait until all have reached a common
    point and can be reused. It employs two semaphores to manage the two phases
    of synchronization, ensuring proper reentrancy.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the Reusable Semaphore Barrier.

        @param num_threads: The total number of threads that will participate in this barrier.
        """
        self.num_threads = num_threads
        # Count of threads for the first phase.
        self.count_threads1 = self.num_threads
        # Count of threads for the second phase.
        self.count_threads2 = self.num_threads
        # Lock to protect access to thread counters.
        self.counter_lock = Lock()
        # Semaphore for the first phase of the barrier.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second phase of the barrier.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all
               participating threads have called `wait()`.
        """
        # Block Logic: Executes the first phase of the barrier synchronization.
        self.phase1()
        # Block Logic: Executes the second phase of the barrier synchronization.
        self.phase2()

    def phase1(self):
        """
        @brief Implements the first phase of the two-phase semaphore barrier.

        Threads decrement a counter. The last thread to reach zero releases
        all other waiting threads for this phase.
        """
        with self.counter_lock:
            # Decrement the count of threads for the first phase.
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Block Logic: If this is the last thread, release all `num_threads` waiting on `threads_sem1`.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the count for `count_threads1` for the next use of the barrier.
                self.count_threads1 = self.num_threads

        # Block Logic: All threads acquire from `threads_sem1`, ensuring they wait until all are ready.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Implements the second phase of the two-phase semaphore barrier.

        Threads decrement a second counter. The last thread to reach zero releases
        all other waiting threads for this phase. This ensures the barrier is reentrant.
        """
        with self.counter_lock:
            # Decrement the count of threads for the second phase.
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Block Logic: If this is the last thread, release all `num_threads` waiting on `threads_sem2`.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the count for `count_threads2` for the next use of the barrier.
                self.count_threads2 = self.num_threads

        # Block Logic: All threads acquire from `threads_sem2`, ensuring they wait until all are ready for the second phase.
        self.threads_sem2.acquire()
