"""
@96f6759b-2cb3-480a-bfc4-4ca45681e3e8/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices using a custom reentrant semaphore barrier and dynamic location-based locking.

This module defines the core components for simulating a network of sensor devices.
It features a `ReusableBarrier` (semaphore-based) implemented directly within the module
for efficient synchronization. Each `Device` operates with a main `DeviceThread` that
manages `RunScripts` worker threads. These workers execute individual scripts,
managing local sensor data and interacting with neighbors. A key aspect is the
dynamic sharing and creation of `Lock` objects for `location_lock` across devices
to ensure thread-safe access to location-specific data.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate under the guidance of a supervisor.

Classes:
- ReusableBarrier: A custom reentrant barrier implementation using semaphores.
- RunScripts: A worker thread responsible for executing individual scripts for specific locations.
- Device: Represents a single simulated sensor device, orchestrating its workers and synchronization.
- DeviceThread: The main thread for a device, managing timepoint progression and worker threads.

Domain: Concurrent Programming, Distributed Systems Simulation, Parallel Processing, Custom Synchronization Primitives.
"""

from threading import Thread, Event
from threading import Lock, Semaphore

class ReusableBarrier():
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

class RunScripts(Thread):                                         
    """
    @brief A worker thread responsible for executing individual scripts for specific locations.

    Each `RunScripts` thread processes a single script, collects data from neighbors,
    executes the script, and updates relevant data in a thread-safe manner
    using location-specific locks.
    """
    
    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes a RunScripts worker thread.

        @param device: The parent `Device` instance.
        @param location: The specific location for which the script will be executed.
        @param script: The script object to be executed.
        @param neighbours: A list of neighboring `Device` instances to interact with.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    
    def run(self):
        """
        @brief The main execution method for the RunScripts thread.

        Pre-condition: `script` is valid, and device/neighbor references are valid.
        Invariant: The thread executes the script, collecting and updating data,
                   while ensuring thread-safe access to location-specific resources.
        """
        # Critical Section: Acquire the location-specific lock to ensure exclusive access
        # to data for this `location` during data collection, script execution, and data updates.
        self.device.location_lock[self.location].acquire()

        script_data = [] # List to store data collected for the script.
        
        # Block Logic: Gathers data from neighboring devices for the current `location`.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Gathers data from its own sensor_data for the current `location`.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Executes the script with the collected data.
            result = self.script.run(script_data)
            
            # Block Logic: Updates data on neighboring devices with the script's `result`.
            for device in self.neighbours:
                device.set_data(self.location, result)
                
            # Updates its own data with the script's `result`.
            self.device.set_data(self.location, result)

        # Releases the location-specific lock after computation and updates are done.
        self.device.location_lock[self.location].release()

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
        # Event to signal that a script has been received for the current timepoint (appears unused).
        self.script_received = Event()
        # List to store assigned scripts, each being a tuple of (script, location).
        self.scripts = []
        # List of all devices in the simulation (populated during setup_devices).
        self.devices = []
        # Event to signal that all scripts for the current timepoint have been assigned.
        self.timepoint_done = Event()

        # The main thread responsible for the device's lifecycle.
        self.thread = DeviceThread(self)
        # Global barrier for inter-device synchronization.
        self.barrier = None
        # List to hold references to spawned `RunScripts` worker threads.
        self.list_thread = []
        self.thread.start()
        # Array of Locks (`location_lock`) for location-specific data protection, initialized to None.
        self.location_lock = [None] * 200 # Assumes a maximum of 200 locations.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up global synchronization resources for all devices.

        This method initializes a global `ReusableBarrier` and manages the distribution
        of `Lock` objects for `location_lock` across devices. The barrier is created
        once and shared among all devices.

        @param devices: A list of all Device instances in the simulation.
        """
        nr_devices = len(devices)
        
        # Block Logic: Initializes the global `ReusableBarrier` if not already set.
        # This ensures a single barrier instance is shared among all devices.
        if self.barrier is None:
            barrier = ReusableBarrier(nr_devices)
            self.barrier = barrier
            # Distributes the global barrier to all other devices.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        # Block Logic: Populates the `self.devices` list with all participating devices.
        for device in devices:
            if device is not None:
                self.devices.append(device)


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed for a specific location on this device.

        This method handles adding the script to the device's pending list and
        managing the `location_lock` for the given location, either creating
        a new `Lock` or acquiring a reference to an existing one from another device.
        If `script` is None, it signals that all scripts for the current timepoint
        have been assigned.

        @param script: The script object to be executed, or None to signal completion.
        @param location: The location pertinent to the script execution.
        """
        lock_location = False # Flag to track if a lock for the location has been found/created.

        if script is None:
            # Pre-condition: `script` is None, signaling end of script assignment for this timepoint.
            # Invariant: The `timepoint_done` event is set, signaling workers to begin processing.
            self.timepoint_done.set()
        else:
            # Pre-condition: `script` is not None.
            # Invariant: The script is added to the device's list of pending scripts.
            self.scripts.append((script, location))
            # Block Logic: Manages the `location_lock` for the current `location`.
            # If a lock for this location doesn't exist locally, it attempts to find one
            # from another device; otherwise, it creates a new Lock.
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        # Inline: If a lock exists on another device, share its reference.
                        self.location_lock[location] = device.location_lock[location]
                        lock_location = True
                        break

                if lock_location is False:
                    # Inline: If no existing lock is found, create a new one for this location.
                    self.location_lock[location] = Lock()

            self.script_received.set() # Signals that a script has been received (appears unused).
            

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
    and spawns `RunScripts` workers to execute scripts. It also handles
    inter-device synchronization using a global barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        @param device: The `Device` instance that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
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
            index = 0 # Starting index for distributing scripts (its value is always 0 here).

            # Block Logic: Copies scripts from the device's script list to a local list.
            for script in self.device.scripts:
                script_list.append(script)
            
            # Block Logic: Spawns 8 `RunScripts` workers.
            # This implementation spawns 8 threads, but currently each is given the same `index=0`,
            # which will cause all 8 threads to try and process `script_list[0]` and then `pop(0)`
            # from the same shared list, leading to `IndexError` after the first thread.
            # This indicates a potential bug in task distribution.
            for i in xrange(8):
                thread = RunScripts(self.device, script_list, neighbours, index) # Pass index 0 repeatedly.
                thread.start()
                thread_list.append(thread)
            
            # Block Logic: Waits for all spawned worker threads to complete their execution.
            for i in xrange(len(thread_list)):
                thread_list[i].join()

            # Clears the list of `RunScripts` threads for the next timepoint.
            self.device.list_thread = [] # This line doesn't affect `thread_list` which is local.
            # Clears `timepoint_done` for the next timepoint (redundant as it's cleared at loop start).
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes with all other DeviceThreads using the global barrier.
            # This ensures all devices have finished processing their scripts before proceeding.
            self.device.barrier.wait()