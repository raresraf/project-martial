"""
@file device.py
@brief Implements a simulated distributed device system featuring a two-phase semaphore-based reusable barrier and dynamically created per-location data locks.

This module provides the framework for simulating a network of devices that can
execute scripts concurrently across different data locations. It employs a
`ReusableBarrierSem` for synchronizing time-step progression among all
participating devices and uses `Lock` objects, dynamically allocated per data
location, to ensure thread-safe access and modification.
"""

from threading import Lock, Thread, Event, Semaphore


class Device(object):
    """
    @class Device
    @brief Represents an individual device in the simulated distributed system.

    Each device manages its own sensor data, processes assigned scripts, and
    coordinates with a central supervisor. It utilizes an `Event` to signal
    script readiness, and relies on a shared `ReusableBarrierSem` for time-step
    synchronization and a shared dictionary of `Lock` objects (`location_locks`)
    for fine-grained, per-location data protection.
    """
    # Invariant: A class-level variable to store the total number of devices, initialized during setup.
    devices_no = 0

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
        # Invariant: An Event to signal that new scripts have been assigned to the device.
        self.script_received = Event()
        # Invariant: A list to store assigned scripts, each paired with its target location.
        self.scripts = []
        # Invariant: An Event to signal that all scripts for the current timepoint have been assigned.
        self.timepoint_done = Event()
        # Invariant: The dedicated thread responsible for this device's operational logic.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Invariant: Reference to the shared ReusableBarrierSem instance for time-step synchronization.
        self.barrier = None
        # Invariant: Reference to the shared dictionary of Locks, protecting data access per location.
        self.location_locks = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs initial setup for the device, including initializing and distributing
        the shared `ReusableBarrierSem` and the shared `location_locks` dictionary.

        Block Logic:
        1. Sets the class-level `devices_no` to the total count of devices.
        2. Only `device_id == 0` initializes a new `ReusableBarrierSem` and an empty `location_locks` dictionary.
        3. All other devices then inherit these shared objects from `devices[0]`.
        @param devices A list of all Device objects in the simulation.
        """
        Device.devices_no = len(devices) # Inline: Sets the total number of devices in the simulation.
        # Block Logic: Device with ID 0 is responsible for initializing shared resources.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices)) # Inline: Initializes the shared barrier.
            self.location_locks = {} # Inline: Initializes the shared dictionary for location-specific locks.
        else:
            # Block Logic: Other devices inherit the shared barrier and location locks from device 0.
            self.barrier = devices[0].barrier
            self.location_locks = devices[0].location_locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specified location.
        If `script` is `None`, it signals that all scripts for the current timepoint have been assigned.
        @param script The script object to be executed, or `None`.
        @param location The data location associated with the script.
        """
        # Block Logic: Appends the script to the internal list and sets `script_received`, or sets `timepoint_done`.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: The appropriate per-location lock from `self.location_locks` must be acquired
        by the calling `run_scripts` thread before this method is called.
        @param location The key for the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        # Pre-condition: Checks if the requested location exists in `sensor_data`.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.
        Note: The appropriate per-location lock from `self.location_locks` must be held
        by the calling `run_scripts` thread.
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
    2. Waiting for `timepoint_done` to be signaled, indicating all scripts are assigned.
    3. Dispatching `run_scripts` worker threads for each assigned script concurrently.
    4. Waiting for all worker threads to complete their tasks for the current timepoint.
    5. Clearing `timepoint_done` for the next cycle.
    6. Synchronizing with all other DeviceThreads via the `ReusableBarrierSem`.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device object this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_scripts(self, script, location, neighbours):
        """
        @brief This method serves as the target function for worker threads dispatched by DeviceThread.

        Block Logic:
        1. Ensures a `Lock` object exists for the given `location` in `device.location_locks`, creating it if necessary.
        2. Acquires the `Lock` for the `location`.
        3. Collects data from neighboring devices and its own device.
        4. Executes the script with the collected data.
        5. Updates the data on neighboring devices and its own device with the script's result.
        6. Releases the `Lock` for the `location`.
        @param script The script object to be executed.
        @param location The specific data location to process.
        @param neighbours A list of neighboring Device objects.
        """
        # Block Logic: Dynamically creates a `Lock` for `location` if it doesn't already exist.
        lock_location = self.device.location_locks.get(location)
        if lock_location is None and location is not None:
            self.device.location_locks[location] = Lock()
            lock_location = self.device.location_locks[location]

        lock_location.acquire() # Block Logic: Acquires the specific Lock for `location`.
        script_data = [] # Invariant: List to store collected data for the script.

        # Block Logic: Collects data from neighboring devices for the specified location.
        for device in neighbours:
            data = device.get_data(location)
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
            for device in neighbours:
                device.set_data(location, result)

            # Block Logic: Updates the data on its own device with the script's result.
            self.device.set_data(location, result)
        lock_location.release() # Inline: Releases the specific Lock for `location`.

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

            # Block Logic: Waits until the `timepoint_done` event is set, indicating all scripts are assigned.
            self.device.timepoint_done.wait()
            tlist = [] # Invariant: List to hold worker thread instances for the current time-step.

            # Block Logic: Creates and starts a worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                # Inline: Creates a new Thread that will execute `self.run_scripts`.
                thread = Thread(target=self.run_scripts, args=(script, location, neighbours))
                tlist.append(thread)
                thread.start()

            # Block Logic: Waits for all worker threads to complete their execution for this timepoint.
            for thread in tlist:
                thread.join()

            # Block Logic: Clears the `timepoint_done` event, preparing for the next round of script assignments.
            self.device.timepoint_done.clear()
            # Block Logic: All devices synchronize at the shared `ReusableBarrierSem`, ensuring all devices
            # have completed their local processing for the current time-step before proceeding.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    @class ReusableBarrierSem
    @brief A reusable, two-phase synchronization barrier implemented using semaphores and a lock.

    Algorithm: This barrier operates in two alternating phases (`phase1` and `phase2`).
    In each phase, threads decrement a counter protected by a lock. The last thread to
    decrement the counter to zero releases all waiting threads using a semaphore and
    resets the counter. Other threads wait on their respective semaphore. This design
    allows the barrier to be reused reliably for multiple synchronization points.
    Time Complexity:
        - `wait()`: Each call involves two phases. The "release" part of a phase is O(N)
          (where N is num_threads) due to multiple `semaphore.release()` calls.
          The "acquire" part is O(1).
    Space Complexity: O(1) for counters, locks, and semaphores.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrierSem with a specified number of participating threads.
        @param num_threads The total number of threads that must reach the barrier before it releases.
        """
        self.num_threads = num_threads
        # Invariant: `count_threads1` tracks threads remaining in the first phase.
        self.count_threads1 = self.num_threads
        # Invariant: `count_threads2` tracks threads remaining in the second phase.
        self.count_threads2 = self.num_threads

        # Invariant: `counter_lock` protects access to the shared counters (`count_threads1`, `count_threads2`).
        self.counter_lock = Lock()
        # Invariant: `threads_sem1` is used to block and release threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        # Invariant: `threads_sem2` is used to block and release threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have completed both phases of the barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Manages the first synchronization phase of the barrier.
        """
        # Block Logic: Decrements the thread counter and checks if all threads have arrived.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Pre-condition: `self.count_threads1` is 0, meaning all threads have reached this phase.
            if self.count_threads1 == 0:
                # Block Logic: Releases all waiting threads and resets the counter for reuse.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        # Block Logic: Blocks the current thread until the semaphore is released by the last arriving thread.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Manages the second synchronization phase of the barrier.
        """
        # Block Logic: Decrements the thread counter and checks if all threads have arrived.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Pre-condition: `self.count_threads2` is 0, meaning all threads have reached this phase.
            if self.count_threads2 == 0:
                # Block Logic: Releases all waiting threads and resets the counter for reuse.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        # Block Logic: Blocks the current thread until the semaphore is released by the last arriving thread.
        self.threads_sem2.acquire()
