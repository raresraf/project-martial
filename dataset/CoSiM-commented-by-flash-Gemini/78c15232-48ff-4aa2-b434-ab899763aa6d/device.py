"""
@78c15232-48ff-4aa2-b434-ab899763aa6d/device.py
@brief Implements a simulated device for a distributed sensor network, with concurrent script execution and a custom reusable barrier.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` for operational logic and uses a `ReusableBarrier`
implemented with `threading.Condition` for global time-step synchronization.
Data access is protected by a pre-allocated list of `Lock` objects (`locationLock`)
on a per-location basis.
"""

from threading import *
import Queue # Note: Queue module is deprecated in Python 3, replaced by queue.

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
        self.count_threads -= 1;
        if self.count_threads == 0:
            self.cond.notify_all() # Last thread to arrive notifies all waiting threads.
            self.count_threads = self.num_threads # Reset counter for next reuse.
        else:
            self.cond.wait(); # Threads wait here until notified by the last thread.
        self.cond.release(); # Release the condition lock.

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, a shared barrier, and a list of location-specific locks.
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
        self.barrier = None # Shared ReusableBarrier for global time step synchronization.
        self.locationLock = None # List of Locks for location-specific data access.
        self.thread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (global barrier and location-specific locks) among all devices.
        Only the device with `device_id == 0` is responsible for initializing these resources,
        which are then distributed to all other devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: The device with `device_id == 0` initializes the shared barrier and `locationLock`.
        if self.device_id == 0:

            # Block Logic: Initializes the shared `ReusableBarrier` with the total number of devices.
            self.barrier = ReusableBarrier(len(devices))

            # Block Logic: Initializes a list of Locks for a large fixed number of potential locations.
            # This list will be shared among all devices.
            self.locationLock = []
            for i in range(0, 10000): # Pre-allocating a large number of locks.
                loc = Lock()
                self.locationLock.append(loc)

            # Block Logic: Distributes the initialized shared `barrier` and `locationLock` to all devices.
            for i in devices:
                i.barrier = self.barrier
                i.locationLock = self.locationLock

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Signals that a script has been received, or that a timepoint is done if no script.
        @param script: The script object to assign.
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
        Note: This method does not acquire any locks directly. It is expected that the calling
        `DeviceThread` (via `run_script` method) will acquire the appropriate `locationLock`
        before calling this method.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `DeviceThread` (via `run_script` method) will acquire the appropriate `locationLock`
        before calling this method.
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


class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts concurrently via dynamically created threads, and coordinating
    with other device threads using a shared `ReusableBarrier`.
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

    def run_script(self, script, location, neighbours):
        """
        @brief Executes a single script for a given location, collecting data from neighbors and itself.
        This method acquires and releases a lock specific to the `location` to ensure atomic updates
        for that data point.
        @param script: The script object to execute.
        @param location: The data location the script operates on.
        @param neighbours: A list of neighboring `Device` instances.
        """
        script_data = []
        # Block Logic: Acquires the location-specific lock from the shared `locationLock` list.
        with self.device.locationLock[location]:

            # Block Logic: Collects data from neighboring devices for the specified location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: Collects data from its own device for the specified location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)


            # Block Logic: Executes the script if any data was collected and propagates the result.
            if script_data != []:
                
                result = script.run(script_data)

                # Block Logic: Updates neighboring devices with the script's result.
                for device in neighbours:
                    device.set_data(location, result)

                # Block Logic: Updates its own device's data with the script's result.
                self.device.set_data(location, result)


    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.
        3. Creates and starts a new `Thread` for each script, using `run_script` as the target function.
           Invariant: All scripts for the current timepoint are executed concurrently.
        4. Waits for all these dynamically created threads to complete.
        5. Synchronizes with all other device threads using a shared `ReusableBarrier`.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        6. Clears `script_received` and `timepoint_done` events for the next cycle.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()

            # Block Logic: Creates a list of tuples representing tasks to be executed.
            queue = [] # Renamed from original for clarity, represents tasks for threads.
            # Block Logic: Populates the queue with script execution tasks.
            for (script, location) in self.device.scripts:
                queue.append((script, location, neighbours))

            subThList = [] # List to keep track of dynamically created threads.
            # Block Logic: Creates and appends new `Thread` instances for each task in the queue.
            # Each thread targets the `run_script` method for concurrent execution.
            while len(queue) > 0:
                # Note: `Queue.pop()` is not a standard list method. Assuming it means `list.pop()` or similar.
                subThList.append(Thread(target = self.run_script, args = queue.pop()))

            # Block Logic: Starts all dynamically created script execution threads.
            for t in subThList:
                t.start()

            # Block Logic: Waits for all initiated script execution threads to complete.
            for t in subThList:
                t.join()

            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.barrier.wait()

            # Block Logic: Clears `script_received` and `timepoint_done` events for the next timepoint cycle.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
