"""
@file device.py
@brief Implements a simulated distributed device system with concurrent script execution and synchronized time-step progression.

This module defines the core components for simulating a network of devices.
It features a shared, reusable barrier for global synchronization across devices,
and each device manages its own pool of worker threads to execute scripts.
Data access is protected by granular locks associated with specific data locations.
"""

from threading import Event, Thread, RLock, Lock, Semaphore, Condition


class Device(object):
    """
    @class Device
    @brief Represents an individual device within the simulated distributed system.

    Each device maintains its sensor data, receives and processes scripts,
    and coordinates its operations with a central supervisor and other devices.
    It employs a static `ReusableBarrier` for system-wide synchronization and
    manages script execution using an `RLock` and a `Semaphore`.
    """
    # Invariant: A static (class-level) reference to the shared ReusableBarrier instance.
    # This barrier is initialized by the first device (device_id == 0) and shared among all.
    barrier = None

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
        # Invariant: An RLock to protect access to the `scripts` list and to coordinate script assignment.
        self.run_script = RLock()
        # Invariant: A Semaphore to limit the number of concurrently running `MyThread` workers.
        self.scripts_sem = Semaphore(8)

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs initial setup for the device, primarily initializing the shared barrier.
        Only the device with `device_id == 0` initializes the global `ReusableBarrier`.
        @param devices A list of all Device objects in the simulation (not directly used by this method, but passed for context).
        """
        # Block Logic: Initializes the static `Device.barrier` only once by the first device created.
        # This ensures all devices share the same barrier for synchronization.
        if Device.barrier is None and self.device_id == 0:
            Device.barrier = ReusableBarrier(len(devices))

    def assign_script(self, script, location):
        """
        @brief Assigns a script and its target location to the device for future execution.
        This method uses an RLock to ensure thread-safe modification of the `scripts` list
        and setting of the `script_received` and `timepoint_done` events.
        @param script The script object to execute, or `None` to signal timepoint completion.
        @param location The data location relevant to the script execution.
        """
        self.run_script.acquire() # Block Logic: Acquire RLock to protect critical section.
        self.script_received.set() # Inline: Signals that a script has been received (or a timepoint is being signaled).
        # Block Logic: Adds the script to the list or marks timepoint completion based on `script` parameter.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
        self.run_script.release() # Inline: Release RLock.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire a lock specific to the location, as locking is handled by `MyThread`.
        @param location The key for the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        # Pre-condition: Checks if the requested location exists in `sensor_data`.
        if location in self.sensor_data:
            return self.sensor_data[location]
        # Invariant: Returns None if the location is not found.
        return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.
        Note: This method does not release a lock specific to the location, as locking is handled by `MyThread`.
        @param location The key for the sensor data.
        @param data The new data to set.
        """
        # Pre-condition: Checks if the requested location exists in `sensor_data` before updating.
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
    @brief The dedicated operational thread for a Device object.

    This thread manages the device's main lifecycle, including:
    1. Periodically checking for neighbor information from the supervisor.
    2. Dispatching `MyThread` workers to execute assigned scripts concurrently.
    3. Waiting for all worker threads to complete their tasks.
    4. Synchronizing with other DeviceThreads using the global `ReusableBarrier`.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device object this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Invariant: The loop continues as long as the supervisor provides neighbor information.
        Each iteration represents a time-step in the simulation.
        """
        while True:
            # Block Logic: Retrieves current neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If `neighbours` is None, it signifies a shutdown or termination condition.
            if neighbours is None:
                break

            # Block Logic: Waits until `script_received` event is set, indicating scripts are available.
            self.device.script_received.wait()

            # Block Logic: Acquires RLock to protect the `scripts` list during worker dispatch.
            self.device.run_script.acquire()
            # Invariant: A dictionary to keep track of worker threads for joining later.
            dictionar = {}
            i = 0
            # Block Logic: Iterates through assigned scripts, creating and starting a worker thread for each.
            for (script, location) in self.device.scripts:
                # Block Logic: Acquires a semaphore, limiting the number of concurrent worker threads.
                self.device.scripts_sem.acquire()
                thread = MyThread(self.device, neighbours, location, script)
                dictionar[i] = thread
                dictionar[i].start()
                i = i + 1
            self.device.run_script.release() # Inline: Releases RLock after all workers are dispatched.
            # Block Logic: Waits for all dispatched worker threads to complete their execution.
            for idx in range(0, len(dictionar)):
                dictionar[idx].join()

            # Block Logic: All DeviceThreads synchronize at the global barrier, ensuring all devices
            # have completed their local processing for the current time-step.
            Device.barrier.wait()
            # Block Logic: Waits until the `timepoint_done` event is set, signaling the end of script assignment for this timepoint.
            self.device.timepoint_done.wait()


class MyThread(Thread):
    """
    @class MyThread
    @brief A worker thread responsible for executing a single script for a specific data location.

    This thread gathers data from its device and neighboring devices, runs the
    assigned script, and then updates the relevant data. It ensures data consistency
    by using per-location locks (`lockForLocations`).
    """
    # Invariant: A static dictionary to hold Lock objects, ensuring that each data location
    # has a unique lock that can be acquired by any MyThread instance.
    lockForLocations = {}

    def __init__(self, device, neighbours, location, script):
        """
        @brief Initializes a MyThread worker.
        @param device The Device object this worker belongs to.
        @param neighbours A list of neighboring Device objects.
        @param location The specific data location this worker is processing.
        @param script The script object to be executed.
        """
        Thread.__init__(self)
        self.location = location
        self.script = script
        self.device = device
        self.neighbours = neighbours

        # Block Logic: Ensures that a Lock exists for the specific data `location`, creating it if necessary.
        if location not in MyThread.lockForLocations:
            MyThread.lockForLocations[location] = Lock()

    def run(self):
        """
        @brief The main execution logic for the MyThread worker.

        Block Logic: Acquires the lock for its assigned data location, collects data
        from its own device and neighbors, executes the script, updates the data,
        releases the lock, and signals completion to the device's semaphore.
        """
        # Block Logic: Acquires the specific lock for `self.location` to ensure exclusive access to data.
        MyThread.lockForLocations[self.location].acquire()
        script_data = []

        # Block Logic: Collects data from neighboring devices for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            # Invariant: Only appends data if it's not None.
            if data is not None:
                script_data.append(data)

        # Block Logic: Collects data from its own device for the specified location.
        data = self.device.get_data(self.location)
        # Invariant: Only appends data if it's not None.
        if data is not None:
            script_data.append(data)

        # Pre-condition: If `script_data` is not empty, there is data to process.
        if script_data != []:
            # Block Logic: Executes the script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Updates the data on neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.location, result)

            # Block Logic: Updates the data on its own device with the script's result.
            self.device.set_data(self.location, result)
        MyThread.lockForLocations[self.location].release() # Inline: Releases the lock for `self.location`.
        self.device.scripts_sem.release() # Inline: Releases a semaphore count, allowing another worker thread to start.


class ReusableBarrier(object):
    """
    @class ReusableBarrier
    @brief A synchronization barrier that can be used multiple times by a fixed number of threads.

    Algorithm: This barrier uses a `Condition` object to block threads. Threads decrement
    a shared counter. The last thread to arrive (when the counter reaches zero)
    notifies all waiting threads and resets the counter for future reuse. Other threads
    wait on the condition variable.
    Time Complexity: `wait()` operation is typically O(1) for most threads (they just wait).
                     The last thread's notification can be considered O(N) where N is num_threads,
                     as it potentially wakes up N threads.
    Space Complexity: O(1).
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier with a specified number of participating threads.
        @param num_threads The total number of threads that must reach the barrier before it releases.
        """
        self.num_threads = num_threads
        # Invariant: `count_threads` tracks the number of threads yet to reach the barrier.
        self.count_threads = self.num_threads
        # Invariant: A Condition object used for thread waiting and notification.
        self.cond = Condition()

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have called this method.
        The barrier is reusable, meaning it can be used again after all threads have passed.
        """
        # Block Logic: Acquires the condition variable's lock to protect the shared `count_threads`.
        self.cond.acquire()
        self.count_threads -= 1 # Inline: Decrements the count of threads yet to arrive.
        # Pre-condition: `count_threads` is 0, meaning the current thread is the last to arrive.
        if self.count_threads == 0:
            # Block Logic: Notifies all waiting threads and resets the barrier for reuse.
            self.cond.notify_all()
            self.count_threads = self.num_threads # Inline: Resets the counter.
        else:
            # Block Logic: If not the last thread, waits until notified by the last thread.
            self.cond.wait()
        self.cond.release() # Inline: Releases the condition variable's lock.
