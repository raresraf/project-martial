"""
@6aaf023b-be01-421b-8555-fc3831a7b0e7/device.py
@brief This module implements a distributed simulation or data processing system
       where each device manages its own set of scripts and uses dynamic
       thread spawning for script execution.

It defines three core classes:
- `Device`: Represents a computational node managing sensor data, scripts,
  and its main `DeviceThread`. It also holds the barrier and location-specific locks.
- `DeviceThread`: The primary thread for a `Device`, responsible for fetching
  neighbor data, synchronizing timepoints, and dynamically spawning new threads
  to execute each script.
- `ReusableBarrierSem`: A custom two-phase reusable barrier synchronization
  mechanism utilizing semaphores, ensuring all devices advance simultaneously.

The system relies on `threading` primitives (Lock, Thread, Event, Semaphore)
for concurrency and synchronization, allowing parallel processing of scripts
across multiple devices and within a single device.

Algorithm:
- Decentralized processing: Each `Device` operates semi-autonomously.
- Dynamic worker spawning: For each timepoint, the `DeviceThread` creates a new
  thread for every script to be executed.
- Timepoint synchronization: `DeviceThread`s synchronize at discrete timepoints using `ReusableBarrierSem`.
- Distributed locking: Location-specific locks ensure data consistency across devices
  when scripts modify shared data.
- Data gathering and propagation: Scripts executed by dynamically spawned threads
  gather data from neighbors and update both the device's and neighbors' data.

Time Complexity:
- `Device.__init__`: O(1)
- `Device.setup_devices`: O(D) where D is the total number of devices (only executed by device_id 0), or O(1) otherwise.
- `DeviceThread.run`: O(T * (N_neighbors + S * (C_script + C_lock))) where T is number of timepoints,
                      S is number of scripts per device per timepoint, C_script is script execution cost,
                      C_lock is lock acquisition/release cost.
- `ReusableBarrierSem.__init__`: O(1)
- `ReusableBarrierSem.wait`: O(num_threads) due to semaphore releases.

Space Complexity:
- `Device`: O(S_max) for scripts, O(L) for location locks.
- `DeviceThread`: O(N_neighbors) for neighbors, O(S) for `tlist` (threads for scripts).
- `ReusableBarrierSem`: O(1)
"""

from threading import Lock, Thread, Event, Semaphore


class Device(object):
    """
    @brief Represents a computational node in the distributed simulation.
           Manages its sensor data, scripts to be executed, and coordinates
           with a supervisor. It also holds the barrier and location-specific locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique integer identifier for this device.
        @param sensor_data: A dictionary containing initial sensor data, keyed by location.
        @param supervisor: The central supervisor object responsible for orchestrating devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data          # Functional Utility: Stores sensor data for various locations.
        self.supervisor = supervisor
        self.script_received = Event()          # Functional Utility: Signals when new scripts are assigned.
        self.scripts = []                       # Functional Utility: List to store scripts assigned for the current timepoint.
        self.timepoint_done = Event()           # Functional Utility: Event to signal end of timepoint script assignment.
        self.thread = DeviceThread(self)        # Functional Utility: The main thread managing this device's operations.
        self.thread.start()                     # Functional Utility: Starts the main DeviceThread upon initialization.
        self.barrier = None                     # Functional Utility: Reference to the global ReusableBarrierSem.
        self.location_locks = None              # Functional Utility: Dictionary of locks, keyed by location, for data access.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared synchronization primitives (ReusableBarrierSem and
               location-specific locks) across all devices. This setup is typically
               performed by a single designated device (device_id == 0).
        @param devices: A list of all Device instances in the simulation.
        Pre-condition: All Device instances have been initialized.
        Post-condition: All devices share the same `ReusableBarrierSem` instance and location locks.
        """
        # Functional Utility: Stores the total number of devices (class-level attribute, potentially problematic if not handled carefully).
        Device.devices_no = len(devices)
        # Block Logic: Ensures that global setup is performed only once by the device with ID 0.
        if self.device_id == 0:
            # Functional Utility: Creates a global reusable barrier for all DeviceThreads.
            self.barrier = ReusableBarrierSem(len(devices))
            self.location_locks = {} # Invariant: 'location_locks' will store a unique lock for each unique location.
        else:
            # Functional Utility: Other devices retrieve references to the already created global barrier and locks.
            self.barrier = devices[0].barrier
            self.location_locks = devices[0].location_locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be processed by this device at a specific location.
               If `script` is None, it signals that all scripts for the current timepoint
               have been assigned.
        @param script: The script object to execute.
        @param location: The location ID where the script should be executed.
        Pre-condition: Scripts are being assigned for the current timepoint.
        Post-condition: `script` is added to `self.scripts`, or `timepoint_done` event is set.
        """
        # Block Logic: Appends the script and its location to the list of pending scripts.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Functional Utility: Signals that new scripts have been received.
        else:
            # Functional Utility: Signals to the main DeviceThread that all scripts
            # for the current timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.
        @param location: The location ID for which to retrieve data.
        @return: The sensor data for the given location, or None if not present.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specified location.
        @param location: The location ID for which to update data.
        @param data: The new data value to set for the location.
        Pre-condition: `location` exists in `self.sensor_data`.
        Post-condition: `self.sensor_data[location]` is updated with `data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the main DeviceThread.
        Pre-condition: Device is operational.
        Post-condition: The main DeviceThread has terminated.
        """
        self.thread.join() # Functional Utility: Waits for the main DeviceThread to complete its execution.


class DeviceThread(Thread):
    """
    @brief The primary thread for a Device. It orchestrates the simulation
           timepoint by timepoint, fetches neighbor information, and dynamically
           spawns threads to execute scripts for the current timepoint.
    """

    def __init__(self, device):
        """
        @brief Initializes a DeviceThread instance.
        @param device: The parent Device instance that this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Functional Utility: Initializes base Thread with descriptive name.
        self.device = device

    def run_scripts(self, script, location, neighbours):
        """
        @brief Executes a single script for a given location, handling data acquisition,
               lock management, and result propagation. This method is designed to be
               run in a separate thread.
        @param script: The script object to execute.
        @param location: The location ID for which the script is run.
        @param neighbours: A list of neighboring devices to interact with for data.
        Pre-condition: `script` and `location` are valid, and `neighbours` is populated.
        Post-condition: Sensor data at `location` on this device and neighbors might be updated.
        """
        # Block Logic: Retrieves or creates a lock for the current location to ensure data consistency.
        # Invariant: Each location has a dedicated lock to prevent race conditions during data access.
        lock_location = self.device.location_locks.get(location)
        if lock_location is None and location is not None:
            self.device.location_locks[location] = Lock() # Functional Utility: Creates a new lock if one doesn't exist for the location.
            lock_location = self.device.location_locks[location]
        lock_location.acquire() # Functional Utility: Acquires the lock for exclusive access to data at 'location'.

        script_data = [] # Invariant: 'script_data' will accumulate relevant data for the script.
        
        # Block Logic: Gathers sensor data for the current 'location' from neighboring devices.
        # Invariant: `script_data` grows with valid data from neighbors.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
            
        # Block Logic: Gathers sensor data for the current 'location' from its own device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        # Block Logic: Executes the script if data was gathered and propagates results.
        # Pre-condition: `script_data` is not empty.
        # Post-condition: Sensor data in `neighbours` and `self.device` may be updated with `result`.
        if script_data != []:
            result = script.run(script_data) # Functional Utility: Executes the assigned script with collected data.

            # Block Logic: Propagates the script's result to neighboring devices.
            # Invariant: Each neighbor device's sensor data at 'location' is updated with 'result'.
            for device in neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result) # Functional Utility: Updates the current device's own sensor data.
        lock_location.release() # Functional Utility: Releases the lock for the specific location.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
               It continuously fetches neighbor data, waits for timepoint synchronization,
               spawns threads for each assigned script, waits for their completion,
               and then synchronizes with other DeviceThreads via a global barrier.
        Pre-condition: The device is initialized and ready to operate.
        Invariant: The thread continues to loop, processing timepoints, until
                   `neighbours` becomes None (shutdown signal).
        Post-condition: The thread terminates after processing all timepoints
                        and receiving the shutdown signal.
        """
        # Block Logic: Fetches the initial list of neighboring devices from the supervisor.
        # This list is updated at the beginning of each round (timepoint).
        neighbours = self.device.supervisor.get_neighbours()
        # Block Logic: Main loop for processing timepoints (rounds).
        while True:
            # Functional Utility: Updates the device's neighbor list for the current round.
            self.device.neighbours = neighbours
            # Block Logic: Checks if a shutdown signal (None neighbors) has been received.
            # Pre-condition: `neighbours` is updated for the current round.
            # Post-condition: If `neighbours` is None, the thread breaks from the loop and terminates.
            if neighbours is None:
                break
            # Functional Utility: Waits until all scripts for the current timepoint have been assigned
            # by the Device's `assign_script` method.
            self.device.timepoint_done.wait()
            tlist = [] # Functional Utility: List to keep track of dynamically spawned script execution threads.
            # Block Logic: For each script assigned to this device, a new thread is spawned to execute it.
            # Invariant: Each script is executed in its own thread, and all these threads are joined.
            for (script, location) in self.device.scripts:
                # Functional Utility: Creates a new thread targeting `run_scripts` for each script.
                thread = Thread(target=self.run_scripts, args=(script, location, neighbours))
                tlist.append(thread) # Functional Utility: Adds the new thread to the list.
                thread.start()       # Functional Utility: Starts the script execution thread.
            # Block Logic: Waits for all dynamically spawned script execution threads to complete.
            for thread in tlist:
                thread.join()
            # Functional Utility: Clears the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()
            # Functional Utility: Synchronizes with other DeviceThreads via the global barrier,
            # ensuring all devices complete their timepoint processing before advancing.
            self.device.barrier.wait()
            # Block Logic: Fetches updated neighbor information for the next timepoint.
            # Pre-condition: All devices have synchronized at the barrier.
            neighbours = self.device.supervisor.get_neighbours()


class ReusableBarrierSem():
    """
    @brief Implements a two-phase reusable barrier for synchronizing multiple threads.
           It uses semaphores to block and release threads in distinct phases,
           allowing for multiple synchronization points.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that must reach the barrier
                            in each phase before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Functional Utility: Counter for the first phase of the barrier.
        self.count_threads2 = self.num_threads # Functional Utility: Counter for the second phase of the barrier.
        
        self.counter_lock = Lock()              # Functional Utility: Protects access to the shared counters.
        
        self.threads_sem1 = Semaphore(0)        # Functional Utility: Semaphore for the first phase, initialized to 0 (all blocked).
        
        self.threads_sem2 = Semaphore(0)        # Functional Utility: Semaphore for the second phase, initialized to 0 (all blocked).

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all other
               participating threads have completed both phases of the barrier.
        Pre-condition: `num_threads` is correctly set.
        Post-condition: All threads have completed both phases and are released.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Executes the first phase of the barrier synchronization.
        Pre-condition: All threads are entering the first phase.
        Invariant: `count_threads1` accurately tracks threads arriving; `threads_sem1` controls release.
        Post-condition: All threads have passed phase 1, and the counter is reset for this phase.
        """
        # Block Logic: Atomically decrements the counter for phase 1 and manages semaphore releases.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Block Logic: If this is the last thread in phase 1, release all waiting threads.
            if self.count_threads1 == 0:
                # Functional Utility: Releases all threads waiting on 'threads_sem1'.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Functional Utility: Resets the counter for the next use.

        self.threads_sem1.acquire() # Functional Utility: Waits (blocks) until released by the last thread in phase 1.

    def phase2(self):
        """
        @brief Executes the second phase of the barrier synchronization.
        Pre-condition: All threads have successfully completed phase 1.
        Invariant: `count_threads2` accurately tracks threads arriving; `threads_sem2` controls release.
        Post-condition: All threads have passed phase 2, and the counter is reset for this phase.
        """
        # Block Logic: Atomically decrements the counter for phase 2 and manages semaphore releases.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Block Logic: If this is the last thread in phase 2, release all waiting threads.
            if self.count_threads2 == 0:
                # Functional Utility: Releases all threads waiting on 'threads_sem2'.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Functional Utility: Resets the counter for the next use.

        self.threads_sem2.acquire() # Functional Utility: Waits (blocks) until released by the last thread in phase 2.
