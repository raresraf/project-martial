"""
@50863064-a6a0-4697-965c-ee532bf7a656/device.py
@brief Implements a simulated device for a distributed sensor network, with concurrent script execution and a custom reusable barrier.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` that dispatches scripts to `MyThread` instances.
Synchronization is handled by a `ReusableBarrier` implemented with `threading.Condition`.
Data access to `sensor_data` is protected by `Lock` objects, but a potential race condition
exists in `get_data` as it does not acquire the lock.
"""

from threading import Event, Thread, Condition, Lock


class ReusableBarrier():
    """
    @brief Implements a reusable barrier for synchronizing a fixed number of threads using a Condition object.
    This barrier ensures that all participating threads wait at a synchronization point
    until every thread has reached it, after which all are released simultaneously.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that will participate in this barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Counter for threads waiting at the barrier.
        
        self.cond = Condition() # Condition variable for blocking and releasing threads.
        
        
    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this barrier.
        Invariant: All threads are held until `count_threads` reaches zero, then all are notified and proceed.
        """
        self.cond.acquire() # Acquire the condition lock.
        
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all() # Last thread to arrive notifies all waiting threads.
            
            self.count_threads = self.num_threads # Reset counter for next reuse.
            
        else:
            self.cond.wait() # Threads wait here until notified by the last thread.
            
        self.cond.release() # Release the condition lock.

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread and a shared `ReusableBarrier`.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing the device's local sensor readings.
        @param supervisor: The supervisor object responsible for managing the overall simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal that a script has been assigned.
        self.scripts = [] # List to store assigned scripts.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None # Shared barrier for global time step synchronization.
        self.lock = Lock() # Lock to protect access to this device's sensor_data.

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared `ReusableBarrier` for synchronization among all devices.
        Only the device with `device_id == 0` initializes the barrier, which is then
        referenced by all other devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: Initializes the shared barrier if this is the first device (device_id == 0).
        # Invariant: A single `ReusableBarrier` instance is created and shared across all devices.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        else:
            # Block Logic: Other devices reference the barrier initialized by device 0.
            self.barrier = devices[0].barrier
        

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Signals that a script has been received, or that a timepoint is done if no script.
        @param script: The script object to assign, or `None` to signal completion.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Block Logic: Signals completion of the timepoint if no script is assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire `self.lock`, which could lead to race conditions
        if `set_data` is called concurrently by another thread.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        This method acquires and releases `self.lock` to ensure thread-safe modification
        of the device's own sensor data.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its graceful completion.
        """
        self.thread.join()

class MyThread(Thread):
    """
    @brief A dedicated thread for executing a single script for a specific data location.
    This thread is responsible for gathering data, running the script, and then
    propagating the results to relevant devices, ensuring thread-safe access to data
    through global locks.
    """
    
    def __init__(self, neighbours, device, location, script):
        """
        @brief Initializes a `MyThread` instance.
        @param neighbours: A list of neighboring `Device` instances.
        @param device: The parent `Device` instance for which the script is being run.
        @param location: The data location that the script operates on.
        @param script: The script object to execute.
        """
        Thread.__init__(self)
        self.neighbours = neighbours
        self.device = device
        self.location = location
        self.script = script

    def run(self):
        """
        @brief The main execution logic for `MyThread`.
        Block Logic:
        1. Collects data from neighboring devices and its own device for the specified `location`.
        2. Executes the assigned `script` if any data was collected.
        3. Propagates the script's `result` to neighboring devices and its own device,
           acquiring and releasing the `Lock` for each device's sensor data.
        Invariant: All data modification operations for both self and neighbors are protected by locks.
        """
        script_data = []
        # Block Logic: Collects data from neighboring devices for the specified location.
        # Note: `get_data` does not acquire the lock, posing a potential race condition.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collects data from its own device for the specified location.
        # Note: `get_data` does not acquire the lock, posing a potential race condition.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script if any data was collected and propagates the result.
        if script_data != []:
            
            result = self.script.run(script_data)

            # Block Logic: Updates neighboring devices with the script's result, acquiring their respective locks.
            for device in self.neighbours:
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()
            
            # Block Logic: Updates its own device's data with the script's result, acquiring its lock.
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()

class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts concurrently via `MyThread` instances, and coordinating with
    other device threads using a shared `ReusableBarrier`.
    Time Complexity: O(T * S_total * (N * D_access + D_script_run)) where T is the number of timepoints,
    S_total is the total number of scripts executed by the device, N is the number of neighbors,
    D_access is data access time, and D_script_run is script execution time.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.
        3. Clears the `timepoint_done` event for the next cycle.
        4. Creates and starts a `MyThread` for each assigned script, allowing concurrent execution.
           Invariant: All scripts for the current timepoint are executed in parallel.
        5. Waits for all `MyThread` instances to complete.
        6. Synchronizes with all other device threads using a shared `ReusableBarrier`.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()
            # Block Logic: Clears the `timepoint_done` event for the next timepoint cycle.
            self.device.timepoint_done.clear()

            threads = [] # List to keep track of currently running script execution threads.
            
            # Block Logic: Iterates through assigned scripts, creating and starting a new `MyThread` for each.
            # Invariant: Each script is executed concurrently in its own `MyThread`.
            for (script, location) in self.device.scripts:
                t = MyThread(neighbours, self.device, location, script)
                t.start()
                threads.append(t)

            # Block Logic: Waits for all `MyThread` instances to complete their execution.
            for i in range(len(threads)):
                threads[i].join()
            
            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.barrier.wait()
