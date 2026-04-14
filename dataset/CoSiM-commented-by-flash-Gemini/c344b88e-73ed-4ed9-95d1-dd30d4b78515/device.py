"""
@file device.py
@brief Implements a simulated distributed device system featuring a custom reusable barrier and fine-grained per-location data locking.

This module provides the framework for simulating a network of devices that can
execute scripts concurrently across different data locations. It employs a
two-phase `ReusableBarrier` for synchronizing time-step progression among all
participating devices and uses dynamically created `Lock` objects for each
data location to ensure thread-safe access and modification.
"""

from threading import Event, Thread, RLock, Lock, Semaphore


class ReusableBarrier():
    """
    @class ReusableBarrier
    @brief A synchronization barrier that can be used multiple times by a fixed number of threads.

    Algorithm: This barrier uses a two-phase approach with two semaphores and a lock-protected
    counter. Threads decrement a counter, and the last thread to reach zero releases
    all other waiting threads using a semaphore. The two phases allow the barrier
    to be reused without requiring all threads to pass through a full cycle
    before being re-initialized.
    Time Complexity:
        - `wait()`: O(num_threads) for releasing semaphores in the release phase,
          and O(1) for acquiring a semaphore in the waiting phase.
    Space Complexity: O(1) for counters, locks, and semaphores.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier with a specified number of participating threads.
        @param num_threads The total number of threads that must reach the barrier before it releases.
        """
        self.num_threads = num_threads
        # Invariant: `count_threads1` tracks threads in the first phase of the barrier.
        self.count_threads1 = [self.num_threads]
        # Invariant: `count_threads2` tracks threads in the second phase of the barrier.
        self.count_threads2 = [self.num_threads]
        # Invariant: `count_lock` protects access to the shared counters (`count_threads1`, `count_threads2`).
        self.count_lock = Lock()
        # Invariant: `threads_sem1` is used to block and release threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        # Invariant: `threads_sem2` is used to block and release threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have called this method.
        The barrier is reusable, meaning it can be used again after all threads have passed.
        """
        # Block Logic: Executes the first phase of the barrier synchronization.
        self.phase(self.count_threads1, self.threads_sem1)
        # Block Logic: Executes the second phase of the barrier synchronization.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Internal method to manage a single phase of the barrier synchronization.
        @param count_threads A list containing the current count of threads for this phase.
        @param threads_sem The semaphore associated with this phase.
        """
        # Block Logic: Decrements the thread counter and checks if all threads have arrived.
        with self.count_lock:
            count_threads[0] -= 1
            # Pre-condition: `count_threads[0]` is 0, meaning all threads have reached this phase.
            if count_threads[0] == 0:
                # Block Logic: Releases all waiting threads and resets the counter for reuse.
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        # Block Logic: Blocks the current thread until the semaphore is released by the last arriving thread.
        threads_sem.acquire()


class Device(object):
    """
    @class Device
    @brief Represents an individual device in the simulated distributed system.

    Each device manages its sensor data, receives and processes assigned scripts,
    and coordinates with a supervisor and other devices. It uses an `Event` to signal
    script readiness, and relies on a shared `ReusableBarrier` for time-step
    synchronization and a shared dictionary of `Lock` objects (`loc_lock`)
    for fine-grained, per-location data protection.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary holding the device's sensor data (e.g., location -> value).
        @param supervisor The supervisor object, responsible for providing neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Invariant: A list to store assigned scripts, each paired with its target location.
        self.scripts = []
        # Invariant: An Event to signal that all scripts for the current timepoint have been assigned.
        self.last_script = Event()
        # Invariant: The dedicated thread responsible for this device's operational logic.
        self.thread = DeviceThread(self)
        # Invariant: Reference to the shared ReusableBarrier instance for time-step synchronization.
        self.timepoint_done = None
        # Invariant: Reference to the shared dictionary of Locks, protecting data access per location.
        self.loc_lock = None
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs initial setup for the device, including initializing and distributing
        the shared `ReusableBarrier` and the shared `loc_lock` dictionary.

        Block Logic:
        1. If `timepoint_done` is None (i.e., this is the first device to call setup),
           it initializes a new `ReusableBarrier` and an empty `loc_lock` dictionary.
        2. All devices are then assigned the same `ReusableBarrier` instance (`timepoint_done`)
           and the same `loc_lock` dictionary.
        @param devices A list of all Device objects in the simulation.
        """
        if self.timepoint_done is None:
            # Block Logic: Initializes the shared `ReusableBarrier` for all devices.
            barrier = ReusableBarrier(len(devices))
            # Invariant: Initializes a shared dictionary to hold `Lock` objects for each data location.
            dic = {}
            # Block Logic: Assigns the shared barrier and `loc_lock` dictionary to all devices.
            for dev in devices:
                dev.timepoint_done = barrier
                dev.loc_lock = dic

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specified location.
        If `script` is `None`, it signals that all scripts for the current timepoint have been assigned.
        @param script The script object to be executed, or `None`.
        @param location The data location associated with the script.
        """
        # Block Logic: Appends the script to the internal list or sets the `last_script` event.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.last_script.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: The appropriate per-location lock from `self.loc_lock` must be acquired
        by the calling `run_script` thread before this method is called.
        @param location The key for the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        # Pre-condition: Checks if the requested location exists in `sensor_data`.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.
        Note: The appropriate per-location lock from `self.loc_lock` must be held
        by the calling `run_script` thread.
        @param location The key for the sensor data.
        @param data The new data to set.
        """
        # Pre-condition: Checks if the location exists in `sensor_data` before updating.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its dedicated operational thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief The dedicated operational thread for a Device object, orchestrating time-step execution.

    This thread manages the device's main operational loop, including:
    1. Periodically checking for updated neighbor information from the supervisor.
    2. Waiting for `last_script` to be signaled, indicating all scripts for the timepoint are assigned.
    3. Dynamically creating and managing worker threads (each executing `run_script`).
    4. Waiting for all worker threads to complete their tasks for the current timepoint.
    5. Synchronizing with all other DeviceThreads via the shared `ReusableBarrier` (`timepoint_done`).
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device object this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_script(self, lock, neighbours, location, script):
        """
        @brief This method serves as the target function for worker threads dispatched by DeviceThread.

        Block Logic:
        1. Ensures a `Lock` object exists for the given `location` in `device.loc_lock`.
        2. Acquires the `Lock` for the `location`.
        3. Collects data from neighboring devices and its own device.
        4. Executes the script with the collected data.
        5. Updates the data on neighboring devices and its own device with the script's result.
        6. Releases the `Lock` for the `location`.
        @param lock An RLock used to protect the `loc_lock` dictionary itself.
        @param neighbours A list of neighboring Device objects.
        @param location The specific data location to process.
        @param script The script object to be executed.
        """
        lock.acquire() # Block Logic: Acquires the RLock to protect `loc_lock` dictionary access.
        # Pre-condition: If no Lock exists for this location, a new one is created and stored.
        if not (self.device.loc_lock).has_key(location): # Note: .has_key() is deprecated in Python 3, use 'in'.
            self.device.loc_lock[location] = Lock()
        lock.release() # Inline: Releases the RLock.

        # Block Logic: Acquires the specific Lock for `location` from the shared `loc_lock` dictionary.
        self.device.loc_lock.get(location).acquire()
        script_data = [] # Invariant: List to store collected data for the script.

        # Block Logic: Collects data from neighboring devices for the specified location.
        for dev in neighbours:
            data = dev.get_data(location)
            # Invariant: Only appends data if it's not None.
            if data is not None:
                script_data.append(data)

        # Block Logic: Collects data from its own device for the specified location.
        data = self.device.get_data(location)
        # Invariant: Only appends data if it's not None.
        if data is not None:
            script_data.append(data)

        # Pre-condition: If `script_data` is not empty, there is data to process.
        if script_data != []:
            # Block Logic: Executes the script with the collected data.
            result = script.run(script_data)

            # Block Logic: Updates the data on neighboring devices with the script's result.
            for dev in neighbours:
                dev.set_data(location, result)

            # Block Logic: Updates the data on its own device with the script's result.
            self.device.set_data(location, result)

        # Inline: Releases the specific Lock for `location`.
        (self.device.loc_lock.get(location)).release()

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Invariant: The loop continues as long as the supervisor provides neighbor information.
        Each iteration represents a complete time-step in the simulation, from script readiness
        to synchronized completion.
        """
        while True:
            # Block Logic: Retrieves current neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If `neighbours` is None, it signifies a shutdown or termination condition.
            if neighbours is None:
                break

            # Block Logic: Waits until the `last_script` event is set, indicating all scripts are assigned.
            self.device.last_script.wait()

            lock = RLock() # Invariant: An RLock to protect access to the `loc_lock` dictionary.
            threads = [] # Invariant: List to hold worker thread instances for the current time-step.

            # Block Logic: Creates and starts a worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                # Inline: Creates a new Thread that will execute `self.run_script`.
                thread = Thread(target=self.run_script, args=(lock, neighbours, location, script))
                thread.start()
                threads.append(thread)

            # Block Logic: Waits for all worker threads to complete their execution for this timepoint.
            for thread in threads:
                thread.join()

            # Block Logic: All devices synchronize at the shared `ReusableBarrier`, ensuring all devices
            # have completed their local processing for the current time-step before proceeding.
            self.device.timepoint_done.wait()
            # Block Logic: Clears the `last_script` event, preparing for the next round of script assignments.
            self.device.last_script.clear()
