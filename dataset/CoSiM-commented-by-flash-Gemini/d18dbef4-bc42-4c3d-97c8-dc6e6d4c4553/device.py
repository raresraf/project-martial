

"""
@d18dbef4-bc42-4c3d-97c8-dc6e6d4c4553/device.py
@brief Implements a simulated sensor device for a distributed sensor network, including its operational thread and a reusable barrier for synchronization.

This module defines the core components for a simulated device in a distributed sensor network.
The `Device` class represents an individual sensor device, managing its own sensor data,
a queue of scripts to execute, and synchronization primitives. The `DeviceThread` class
provides a dedicated execution context for each `Device`, continuously processing scripts
assigned to it by spawning new threads for each script. The `ReusableBarrierSem` class
implements a two-phase reusable barrier for robust thread synchronization.

Domain: Distributed Systems, Concurrency, Simulation, Sensor Networks.
"""

from threading import Lock, Thread, Event, Semaphore


class Device(object):
    """
    @brief Represents a simulated sensor device in a distributed sensor network.

    This class manages the device's unique identifier, sensor data,
    and a reference to its supervisor. It handles the receipt and
    storage of scripts for execution, along with synchronization
    mechanisms such as an `Event` for script assignment, a list
    to store incoming scripts, and a shared pool of `Lock` objects
    for managing concurrent access to sensor data locations. Each device
    also has a dedicated thread (`DeviceThread`) for operation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        :param device_id: A unique identifier for the device.
        :param sensor_data: A dictionary representing the sensor data
                            collected by this device, where keys are
                            sensor locations and values are data points.
        :param supervisor: A reference to the supervisor object that
                           manages the network of devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Functional Utility: Signals when a new script has been assigned to the device.
        self.script_received = Event()
        # Functional Utility: Stores a list of (script, location) tuples to be executed.
        self.scripts = []
        # Functional Utility: Signals that all scripts for the current time point have been assigned.
        self.timepoint_done = Event()
        # Functional Utility: The dedicated thread for this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Functional Utility: Reference to the shared reusable barrier for inter-device synchronization.
        self.barrier = None
        # Functional Utility: Reference to the shared dictionary of locks for sensor locations.
        self.location_locks = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        :return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (barrier and location locks) for all devices.

        This method ensures that all devices share the same barrier and a common
        set of location-specific locks. The first device (device_id == 0) is
        responsible for initializing these shared resources, which are then
        referenced by all other devices.

        :param devices: A list of all Device instances in the network.
        """
        # Functional Utility: Stores the total number of devices, used by the barrier.
        Device.devices_no = len(devices)
        if self.device_id == 0:
            # Functional Utility: Initializes the shared reusable barrier with the total number of devices.
            self.barrier = ReusableBarrierSem(len(devices))
            # Functional Utility: Initializes the shared dictionary to hold locks for each sensor location.
            self.location_locks = {}
        else:
            # Functional Utility: Non-zero devices reference the shared barrier initialized by device 0.
            self.barrier = devices[0].barrier
            # Functional Utility: Non-zero devices reference the shared location locks initialized by device 0.
            self.location_locks = devices[0].location_locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.

        If a script is provided, it is added to the device's script queue.
        If no script is provided (i.e., `script` is None), it signals
        that script assignment is complete for the current time point,
        allowing the device's thread to proceed.

        :param script: The script object to be executed, or None to signal timepoint completion.
        :param location: The sensor location associated with the script.
        """
        if script is not None:
            # Functional Utility: Appends the new script and its target location to the processing queue.
            self.scripts.append((script, location))
            # Functional Utility: Sets the event to notify the DeviceThread that a script has been assigned.
            self.script_received.set() # This appears to be a vestige from an earlier design, as timepoint_done is used for signaling completion.
        else:
            # Functional Utility: Sets the event to notify the DeviceThread that all scripts for the current time point have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.

        :param location: The sensor location to retrieve data from.
        :return: The sensor data at the given location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.

        :param location: The sensor location to update.
        :param data: The new data value to set at the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its dedicated thread.

        Ensures that the `DeviceThread` completes its execution before the program exits.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief A dedicated thread for a Device to manage its operational lifecycle.

    Each `Device` instance has a `DeviceThread` that continuously
    monitors for assigned scripts, processes these scripts by spawning
    worker threads, and synchronizes with other `DeviceThread`s
    using a shared barrier. This thread ensures that script processing
    is handled asynchronously and in parallel with other devices.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        :param device: The Device instance this thread is associated with.
        """
        # Functional Utility: Calls the base Thread class constructor, setting a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_scripts(self, script, location, neighbours):
        """
        @brief Executes a single script, handling data aggregation and updates with concurrency control.

        This method is designed to be run as a separate thread. It performs the core logic:
        1. Acquires a lock for the specific `location` to prevent race conditions,
           creating the lock if it doesn't already exist for that location.
        2. Gathers relevant sensor data from all neighboring devices and the local device
           for the given `location`.
        3. Executes the `script` using the aggregated data.
        4. Updates the sensor data on all neighboring devices and the local device
           with the result of the script execution.
        5. Releases the `location` lock.

        :param script: The script object to run.
        :param location: The sensor location being processed.
        :param neighbours: The list of neighboring Device instances.
        """
        # Block Logic: Retrieves or creates a lock for the specific sensor location.
        # This ensures that only one worker thread can modify data for a given location at a time.
        lock_location = self.device.location_locks.get(location)
        if lock_location is None and location is not None:
            self.device.location_locks[location] = Lock()
            lock_location = self.device.location_locks[location]

        # Functional Utility: Acquires the lock for the specific sensor location to ensure exclusive access
        # during data aggregation and update, preventing race conditions.
        lock_location.acquire()
        script_data = []

        # Block Logic: Aggregates sensor data from all neighboring devices for the current location.
        # Precondition: `neighbours` is a list of Device objects.
        # Invariant: `script_data` will contain valid sensor data from neighbors if available.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Functional Utility: Aggregates sensor data from the local device for the current location.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data: # Block Logic: Proceeds with script execution only if there is data to process.
            # Functional Utility: Executes the script with the aggregated sensor data.
            result = script.run(script_data)

            # Block Logic: Updates the sensor data on all neighboring devices with the script's result.
            # Invariant: Neighboring devices' data at `location` will reflect the script's outcome.
            for device in neighbours:
                device.set_data(location, result)

            # Functional Utility: Updates the local device's sensor data with the script's result.
            self.device.set_data(location, result)
        # Functional Utility: Releases the lock for the sensor location, allowing other threads to access it.
        lock_location.release()

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        This method continuously:
        1. Retrieves neighbors from the supervisor. If no neighbors, it breaks the loop.
        2. Waits for the `timepoint_done` event, signaling that all scripts for the current
           time point have been assigned.
        3. Spawns a new thread for each assigned script, executing `run_scripts`.
        4. Waits for all these script execution threads to complete.
        5. Clears the `timepoint_done` event.
        6. Synchronizes with other `DeviceThread`s using a shared barrier
           before starting the next cycle of script processing.
        """
        # Block Logic: Main loop for continuous script processing.
        # Precondition: The device is active and connected to a supervisor.
        # Invariant: The device continuously checks for new timepoints, processes their scripts, and synchronizes.
        while True:
            # Functional Utility: Obtains the list of neighboring devices from the supervisor for data exchange.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Block Logic: Termination condition if no neighbors are found, indicating the simulation might be ending.
                break
            # Functional Utility: Pauses execution until all scripts for the current time point are assigned and `timepoint_done` is set.
            self.device.timepoint_done.wait()
            tlist = []
            # Block Logic: Spawns a new thread for each script to be run concurrently.
            # Invariant: `tlist` contains references to all active script execution threads.
            for (script, location) in self.device.scripts:
                thread = Thread(target=self.run_scripts, args=(script, location, neighbours))
                tlist.append(thread)
                thread.start()
            # Block Logic: Waits for all concurrently running script execution threads to complete.
            # Precondition: All script threads have been started.
            # Invariant: All scripts for the current time point are processed before proceeding.
            for thread in tlist:
                thread.join()
            # Functional Utility: Clears the event for the next timepoint.
            self.device.timepoint_done.clear()
            # Functional Utility: Clears the list of processed scripts.
            self.device.scripts = []
            # Functional Utility: Synchronizes all DeviceThreads, ensuring all devices complete a timepoint
            # before moving to the next. This acts as a global checkpoint.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.

    This barrier allows a fixed number of threads to wait for each other
    to reach a common point before any of them can proceed. It is designed
    to be reusable, meaning it can be used multiple times after all threads
    have passed through it. The implementation uses a two-phase approach
    with semaphores to prevent threads from "slipping through" or
    "bypassing" the barrier if they arrive too early for the next cycle.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrierSem instance.

        :param num_threads: The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # Functional Utility: Counter for the first phase of the barrier.
        self.count_threads1 = self.num_threads

        # Functional Utility: Counter for the second phase of the barrier.
        self.count_threads2 = self.num_threads

        # Functional Utility: A lock to protect the internal counters during updates.
        self.counter_lock = Lock()
        # Functional Utility: Semaphore for the first phase of thread waiting.
        self.threads_sem1 = Semaphore(0)
        # Functional Utility: Semaphore for the second phase of thread waiting.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier.

        The thread will block until all `num_threads` threads have called `wait()`.
        This method orchestrates the two-phase synchronization.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief The first phase of the reusable barrier synchronization.

        Threads decrement a counter. The last thread to reach zero
        releases all waiting threads for this phase via a semaphore,
        then resets the counter. All threads then acquire the semaphore
        to proceed to phase 2.
        """
        # Block Logic: Atomically decrements the counter and checks if all threads have arrived.
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Block Logic: The last thread releases all waiting threads for phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Functional Utility: Resets the counter for the next barrier cycle.
                self.count_threads1 = self.num_threads

        # Functional Utility: Threads wait here until all others have reached the barrier's first phase.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief The second phase of the reusable barrier synchronization.

        Similar to phase 1, but uses a different counter and semaphore
        to ensure proper resetting and reusability of the barrier.
        This prevents race conditions where a fast thread might re-enter
        the barrier before all slow threads have exited the previous cycle.
        """
        # Block Logic: Atomically decrements the counter and checks if all threads have arrived.
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Block Logic: The last thread releases all waiting threads for phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Functional Utility: Resets the counter for the next barrier cycle.
                self.count_threads2 = self.num_threads

        # Functional Utility: Threads wait here until all others have reached the barrier's second phase.
        self.threads_sem2.acquire()

