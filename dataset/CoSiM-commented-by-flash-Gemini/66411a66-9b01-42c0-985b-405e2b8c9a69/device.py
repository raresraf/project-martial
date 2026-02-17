
"""
@66411a66-9b01-42c0-985b-405e2b8c9a69/device.py
@brief Implements a simulated device for a distributed sensor network, with concurrent script execution using a thread pool.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` which manages the distribution of scripts to a fixed pool of
`MyThread` instances for parallel execution. Global synchronization is handled by a
shared `ReusableBarrier`, and data updates are protected by per-device `Lock` objects.
"""

from threading import Event, Thread, Lock
from barrier import * # Assumed to contain ReusableBarrier class.


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread and shared barrier.
    """
    
    # Class-level attribute: shared ReusableBarrier for global time step synchronization.
    barrier = ReusableBarrier(0) # Initialized with 0 threads, then re-initialized in setup_devices.

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
        self.lock = Lock() # Per-device lock to protect its own `sensor_data` during updates.

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
        referenced by all other devices through the class-level `Device.barrier`.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: The device with `device_id == 0` initializes the class-level `ReusableBarrier`.
        # This ensures a single barrier instance is shared by all devices.
        if self.device_id == 0:
            Device.barrier = ReusableBarrier(len(devices))

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
        Note: This method does not acquire `self.lock`. It relies on external locking
        mechanisms (e.g., in `DeviceThread`'s result propagation) for consistency when reading.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire `self.lock`. It relies on external locking
        mechanisms (e.g., in `DeviceThread`'s result propagation) for consistency when writing.
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
    distributing script execution among a fixed pool of `MyThread` instances, and coordinating
    with other device threads using a shared `ReusableBarrier`.
    Time Complexity: O(T * S_total * (N + D_access) / P) where T is the number of timepoints,
    S_total is the total number of scripts executed by the device, N is the number of neighbors,
    D_access is data access time, and P is the size of the `MyThread` pool (8 in this case).
    The use of a thread pool helps to parallelize script execution.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = [] # List to hold `MyThread` instances (the thread pool).

    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.
        3. Clears the `timepoint_done` event for the next cycle.
        4. Distributes assigned scripts among a fixed pool of 8 `MyThread` instances.
        5. Starts all `MyThread` instances that have tasks assigned.
        6. Waits for all active `MyThread` instances to complete their execution.
        7. Synchronizes with all other `DeviceThread` instances using the shared `Device.barrier` (Phase 1).
        8. Collects results from the `MyThread` pool and propagates them to neighbors and its own device,
           protecting each device's `set_data` call with its `lock`.
        9. Synchronizes again with all other `DeviceThread` instances using the shared `Device.barrier` (Phase 2).
           Invariant: All active `DeviceThread` instances must complete their data propagation
           before proceeding to the next timepoint, ensuring synchronized advancement of the simulation.
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

            # Block Logic: Initializes a fixed pool of 8 `MyThread` instances for concurrent script execution.
            self.threads = []
            for i in range(8): # Changed xrange to range for Python 3 compatibility.
                self.threads.append(MyThread("{}-{}".format(self.device.device_id, i)))

            # Block Logic: Iterates through assigned scripts, distributing them among the `MyThread` pool.
            # Invariant: Each script's data is gathered and then appended to one of the `MyThread`s.
            i = 0 # Used to cycle through the `MyThread` pool.
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
                    self.threads[i].add_task(script, location, script_data)
                    i = (i+1) % 8 # Cycles through the 8 `MyThread`s.

            # Block Logic: Starts all `MyThread` instances that have tasks assigned.
            for i in range(8): # Changed xrange to range for Python 3 compatibility.
                if self.threads[i].tasks != []:
                    self.threads[i].start()

            # Block Logic: Waits for all active `MyThread` instances to complete their script execution.
            for i in range(8): # Changed xrange to range for Python 3 compatibility.
                if self.threads[i].tasks != []:
                    self.threads[i].join()
            
            # Block Logic: Synchronizes with other device threads using the shared barrier (Phase 1).
            Device.barrier.wait()

            # Block Logic: Iterates through the results from the `MyThread` pool
            # and propagates the results to neighbors and its own device.
            for i in range(8): # Changed xrange to range for Python 3 compatibility.
                if self.threads[i].results != []:
                    for (script, location, result) in self.threads[i].results:
                        # Block Logic: Updates neighboring devices with the script's result, acquiring their respective locks.
                        for device in neighbours:
                            device.lock.acquire()
                            device.set_data(location, result)
                            device.lock.release()
                        
                        # Block Logic: Updates its own device's data with the script's result, acquiring its lock.
                        self.device.lock.acquire()
                        self.device.set_data(location, result)
                        self.device.lock.release()

            # Block Logic: Synchronizes again with other device threads using the shared barrier (Phase 2).
            Device.barrier.wait()


class MyThread(Thread):
    """
    @brief A worker thread class for executing a list of assigned scripts.
    Each `MyThread` instance processes its own list of tasks (scripts) sequentially
    and stores their results. It is designed to be part of a thread pool.
    """

    def __init__(self, id):
        """
        @brief Initializes a `MyThread` instance.
        @param id: A unique identifier for this thread (e.g., "DeviceID-ThreadIndex").
        """
        Thread.__init__(self, name="Device Thread %s" % id)
        self.tasks = [] # List of tasks: `(script_object, location, script_data)`.
        self.results = [] # List of results: `(script_object, location, result)`.

    def add_task(self, script, location, script_data):
        """
        @brief Adds a script execution task to this worker thread's queue.
        @param script: The script object to execute.
        @param location: The data location relevant to the script.
        @param script_data: The input data for the script.
        """
        self.tasks.append((script, location, script_data))

    def clear(self):
        """
        @brief Clears the list of tasks and results for this worker thread, resetting it for reuse.
        """
        self.tasks = []
        self.results = []

    def run(self):
        """
        @brief Executes all tasks (scripts) assigned to this thread sequentially.
        Block Logic:
        Iterates through the assigned tasks, runs each script with its provided data,
        and stores the script object, location, and computed result.
        """
        for (script, location, script_data) in self.tasks:
            self.results.append((script, location, script.run(script_data)))

