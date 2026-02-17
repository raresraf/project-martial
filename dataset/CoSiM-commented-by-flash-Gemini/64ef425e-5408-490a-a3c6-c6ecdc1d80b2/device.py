"""
@64ef425e-5408-490a-a3c6-c6ecdc1d80b2/device.py
@brief Implements a simulated device for a distributed sensor network, utilizing a thread pool for concurrent script execution and barrier synchronization.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` that distributes script execution among a fixed pool of
`CoreThread` instances. Synchronization is handled by a `ReusableBarrierCond` (Condition-based barrier).
Data access to `sensor_data` is not explicitly locked, which could lead to race conditions.
"""

from threading import Event, Thread, Condition

class ReusableBarrierCond(object):
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
        self.cond = Condition()                  # Condition variable for blocking and releasing threads.

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this barrier.
        Invariant: All threads are held until `count_threads` reaches zero, then all are notified and proceed.
        """
        self.cond.acquire()                      # Acquire the condition lock.
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()               # Last thread to arrive notifies all waiting threads.
            self.count_threads = self.num_threads # Reset counter for next reuse.
        else:
            self.cond.wait()                     # Threads wait here until notified by the last thread.
        self.cond.release()                     # Release the condition lock.

class CoreThread(Thread):
    """
    @brief A worker thread responsible for executing a batch of assigned scripts sequentially.
    This thread collects script execution details and their results. It is intended to be
    part of a larger thread pool managed by `DeviceThread`.
    """
    
    def __init__(self):
        """
        @brief Initializes a `CoreThread` instance.
        """
        Thread.__init__(self)
        self.threads = [] # Stores scripts to be executed: `(script_object, location, data_for_script)`.
        self.results = [] # Stores execution results: `(script_object, location, result_of_script)`.

    def append_script(self, script, location, data):
        """
        @brief Appends a script, its location, and associated data to be processed by this thread.
        @param script: The script object to execute.
        @param location: The data location relevant to the script.
        @param data: The input data for the script.
        """
        self.threads.append((script, location, data))

    def run(self):
        """
        @brief Executes all appended scripts sequentially and stores their results.
        Block Logic:
        Iterates through the list of assigned scripts, runs each with its provided data,
        and stores the script object, location, and computed result.
        """
        self.results = [(script, location, script.run(data)) \
        for (script, location, data) in self.threads]


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, sharing a global barrier for synchronization.
    """
    
    # Class-level shared attributes for synchronization across all device instances.
    barrier = ReusableBarrierCond(0) # Shared barrier for global time step synchronization.
    barrier_set = False # Flag to ensure the barrier is initialized only once.

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
        self.scripts = [] # List to store assigned scripts.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.thread = None # The dedicated `DeviceThread` for this device.

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared `ReusableBarrierCond` for synchronization among all devices
        and initializes this device's operational thread.
        The barrier is initialized only once by the first device to call this method.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup for each device.
        """
        # Block Logic: Initializes the shared barrier only once across all devices.
        if not Device.barrier_set:
            Device.barrier = ReusableBarrierCond(len(devices))
            Device.barrier_set = True

        # Block Logic: Creates and starts the dedicated thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: Signals completion of the timepoint if no script is assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire any locks, which could lead to race conditions
        if `sensor_data` is modified concurrently by another thread without external synchronization.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire any locks, which could lead to race conditions
        if `sensor_data` is modified concurrently by another thread without external synchronization.
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
    distributing script execution among a pool of `CoreThread` instances, and coordinating
    with other device threads using a shared `ReusableBarrierCond`.
    Time Complexity: O(T * S_total * (N + D_access) / P) where T is the number of timepoints,
    S_total is the total number of scripts executed by the device, N is the number of neighbors,
    D_access is data access time, and P is the size of the `CoreThread` pool (8 in this case).
    The use of a thread pool helps to parallelize script execution.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = [] # List to hold `CoreThread` instances (the thread pool).

    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.
        3. Clears the `timepoint_done` event for the next cycle.
        4. Distributes assigned scripts among a fixed pool of 8 `CoreThread` instances.
        5. Starts all `CoreThread` instances that have scripts assigned.
        6. Waits for all active `CoreThread` instances to complete their execution.
        7. Synchronizes with all other `DeviceThread` instances using the shared `Device.barrier`.
        8. Collects results from the `CoreThread` pool and propagates them to neighbors and its own device.
        9. Synchronizes again with all other `DeviceThread` instances using the shared `Device.barrier`.
           Invariant: All active `DeviceThread` instances must complete their data propagation
           before proceeding to the next timepoint, ensuring synchronized advancement.
        """
        while True:
            # Block Logic: Fetches neighbor devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()
            # Block Logic: Clears the `timepoint_done` event for the next timepoint cycle.
            self.device.timepoint_done.clear()
            
            # Block Logic: Initializes a fixed pool of 8 `CoreThread` instances.
            self.threads = [CoreThread() for i in range(8)]
            
            count = 0 # Used to cycle through the `CoreThread` pool.
            # Block Logic: Iterates through assigned scripts, distributing them among the `CoreThread` pool.
            # Invariant: Each script's data is gathered and then appended to one of the `CoreThread`s.
            for (script, location) in self.device.scripts:
                script_data = []
                # Block Logic: Collects data from neighboring devices for the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)


                # Block Logic: Collects data from its own device for the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                if script_data != []:
                    self.threads[count].append_script(script, location, script_data)
                    count = (count+1) % 8 # Cycles through the 8 `CoreThread`s.

            # Block Logic: Starts all `CoreThread` instances that have scripts assigned.
            for i in range(8):
                if self.threads[i].threads != []:
                    self.threads[i].start()
            
            # Block Logic: Waits for all active `CoreThread` instances to complete their script execution.
            for i in range(8):
                if self.threads[i].threads != []:
                    self.threads[i].join()

            # Block Logic: Synchronizes with other device threads using the shared barrier,
            # ensuring all devices complete their script execution phase before proceeding.
            Device.barrier.wait()

            # Block Logic: Iterates through the results from the `CoreThread` pool
            # and propagates the results to neighbors and its own device.
            for i in range(8):
                for (script, location, result) in self.threads[i].results:
                    # Block Logic: Updates neighboring devices with the script's result.
                    for device in neighbours:
                        device.set_data(location, result)
                    # Block Logic: Updates its own device's data with the script's result.
                    self.device.set_data(location, result)
            # Block Logic: Synchronizes again with other device threads using the shared barrier,
            # ensuring all data propagation is complete before moving to the next timepoint.
            Device.barrier.wait()
