"""
@a52f9f91-c477-4963-9668-17b45cb4412c/device.py
@brief Implements a simulated device for a distributed sensor network,
       including multithreaded script execution and synchronized data processing.

This module defines the architecture for individual devices in a sensor
network. Each device manages its sensor data, receives and executes scripts,
and communicates with a supervisor and neighboring devices. It leverages
threading primitives like Events, Locks, and Semaphores for concurrent
script processing and a reusable barrier for global synchronization across devices.
"""


from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.

    Each device has a unique ID, stores sensor data, executes assigned scripts,
    and collaborates with other devices for synchronized data processing.
    It manages its own script queue and communicates its state to a supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's sensor readings,
                            keyed by location.
        @param supervisor: A reference to the supervisor object for inter-device
                           communication and network topology information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        # List to store assigned scripts, each being a (script_object, location) tuple.
        self.scripts = []
        # Lock to protect concurrent access to the 'scripts' list.
        self.scripts_lock = Lock()
        # Event to signal when the current timepoint's script processing is done.
        self.timepoint_done = Event()
        # Reference to the shared barrier for device synchronization. Initialized by setup_devices.
        self.barrier = None
        # Dictionary to store locks for protecting access to specific sensor locations.
        self.location_locks = {}
        # The dedicated thread for running device operations (script processing, synchronization).
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the synchronization barrier and location-specific locks for a group of devices.

        This is a static-like method intended to be called once by the supervisor
        or an orchestrator to initialize shared resources among all devices.

        @param devices: A list of all Device objects in the network.
        """
        # Create a single ReusableBarrierCond instance for all devices.
        # This barrier will coordinate the synchronization of all 'len(devices)' threads.
        barrier = ReusableBarrierCond(len(devices))
        # Assign the same barrier instance to all devices in the network.
        for device in devices:
            device.barrier = barrier

        location_locks = {}
        # Block Logic: Initialize a unique lock for each distinct sensor data location
        #              across all devices. This prevents race conditions when multiple
        #              scripts try to access/modify data at the same location.
        for device in devices:
            for location in device.sensor_data:
                if location not in location_locks:
                    location_locks[location] = Lock()

        # Assign the centrally managed location locks to each device.
        for device in devices:
            device.location_locks = location_locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.

        If a script is provided, it's added to the device's script queue. An event
        is set to notify the device's processing thread that new scripts are available.
        If `script` is None, it signals the end of the current timepoint for script assignment.

        @param script: The script object to be executed, or None.
        @param location: The data location relevant to the script.
        """
        # Block Logic: Manages the assignment of new scripts or signals timepoint completion.
        if script is not None:
            self.scripts_lock.acquire() # Acquire lock to protect the 'scripts' list.
            self.scripts.append((script, location))
            self.scripts_lock.release() # Release lock.
            self.script_received.set() # Signal the DeviceThread that new scripts arrived.
        else:
            self.timepoint_done.set() # Signal that no more scripts are coming for this timepoint.
            self.script_received.set() # Still signal, even if script is None, to wake up DeviceThread.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The location for which to retrieve data.
        @return: The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The location at which to set the data.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def process_work(self, script, location, neighbours):
        """
        @brief Processes a single script, gathering data, executing the script,
               and updating sensor data for the local device and its neighbors.

        This method ensures that access to shared sensor data locations is
        thread-safe using `location_locks`.

        @param script: The script object to execute.
        @param location: The sensor data location pertinent to this script.
        @param neighbours: A list of neighboring Device objects from which to collect data.
        """
        # Acquire the lock for the specific sensor location to prevent race conditions.
        self.location_locks[location].acquire()

        script_data = [] # List to hold data collected for the script.

        # Block Logic: Collect data from neighboring devices at the specified location.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Collect data from the local device at the specified location.
        data = self.get_data(location)

        if data is not None:
            script_data.append(data)

        # Block Logic: If any script data was collected, execute the script and update data.
        if script_data:
            result = script.run(script_data) # Execute the script with collected data.

            # Update sensor data for all neighboring devices.
            for device in neighbours:
                device.set_data(location, result)

            # Update sensor data for the local device.
            self.set_data(location, result)

        # Release the lock for the sensor location.
        self.location_locks[location].release()

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its processing thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main processing thread for a Device.

    This thread manages script execution for its associated device. It coordinates
    with worker threads, handles synchronization barriers, fetches neighbors' data,
    and dispatches scripts for processing.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread with a reference to its parent device.

        @param device: The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously orchestrates script processing: it waits for new scripts,
        dispatches them to a pool of worker threads, waits for all work to complete
        for a timepoint, and then synchronizes with other device threads using a barrier.
        The loop terminates if the supervisor signals no more neighbors (end of simulation).
        """
        work_lock = Lock() # Lock to protect access to the 'work_pool'.
        work_pool_empty = Event() # Event to signal when the work pool is empty.
        work_pool_empty.set() # Initially, the work pool is empty.
        work_pool = [] # List to store scripts awaiting processing by workers.
        workers = [] # List to hold worker thread objects.
        workers_number = 7 # Fixed number of worker threads.
        work_available = Semaphore(0) # Semaphore to notify workers that work is available.
        own_work = None # Placeholder for work processed directly by DeviceThread if no workers.

        # Block Logic: Creates and starts a fixed number of worker threads.
        for worker_id in range(1, workers_number + 1):
            workers.append(Worker(worker_id, work_pool, work_available, work_pool_empty, work_lock, self.device))
            workers[worker_id-1].start()

        # Block Logic: Main loop for continuous operation of the DeviceThread.
        while True:
            scripts_ran = [] # List to track scripts already processed in the current timepoint.
            # Get the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: If supervisor returns None for neighbors, it indicates simulation end.
            if neighbours is not None:
                # Convert neighbors to a set for efficient removal of self.
                neighbours = set(neighbours)
                if self.device in neighbours:
                    neighbours.remove(self.device) # A device is not its own neighbor for data collection.

            if neighbours is None:
                # Block Logic: If simulation ends, signal workers to terminate and then join them.
                for i in range(0,7): # Release semaphore multiple times to wake up all workers.
                    work_available.release()
                for worker in workers:
                    worker.join() # Wait for all worker threads to finish.
                break # Exit the main device thread loop.

            # Synchronization Point: Wait for all devices to reach this barrier before proceeding.
            self.device.barrier.wait()

            # Block Logic: Process scripts assigned to the device for the current timepoint.
            while True:
                # Wait until new scripts are assigned or timepoint is done.
                self.device.script_received.wait()
                self.device.script_received.clear() # Clear the event after processing.

                self.device.scripts_lock.acquire() # Acquire lock to protect scripts list.

                # Block Logic: Iterate through newly assigned scripts and dispatch them.
                for (script, location) in self.device.scripts:
                    # Avoid reprocessing scripts already run in this timepoint.
                    if script in scripts_ran:
                        continue

                    scripts_ran.append(script) # Mark script as ran.

                    # If no dedicated work has been assigned for this thread yet, take it.
                    if own_work is None:
                        own_work = (script, location, neighbours)
                    else:
                        # Otherwise, add work to the shared pool for worker threads.
                        work_lock.acquire()
                        work_pool.append((script, location, neighbours))
                        work_pool_empty.clear() # Indicate work pool is not empty.
                        work_available.release() # Signal a worker that work is available.
                        work_lock.release()

                self.device.scripts_lock.release() # Release scripts list lock.

                # Block Logic: Check if all scripts for the current timepoint are processed.
                # Invariant: 'timepoint_done' is set, and all scripts have been dispatched.
                if self.device.timepoint_done.is_set() and len(scripts_ran) == len(self.device.scripts):
                    # Process work that was taken by the DeviceThread itself.
                    if own_work is not None:
                        script, location, neighbours = own_work
                        own_work = None
                        self.device.process_work(script, location, neighbours)

                    # Wait until the work pool is empty (all worker threads have processed their tasks).
                    work_pool_empty.wait()

                    # Ensure all worker threads have completed their current tasks.
                    for worker in workers:
                        worker.work_done.wait()

                    self.device.timepoint_done.clear() # Clear timepoint_done for the next cycle.
                    # Synchronization Point: Wait for all devices to finish their timepoint processing.
                    self.device.barrier.wait()
                    break # Exit the script processing loop for this timepoint.


class Worker(Thread):
    """
    @brief A worker thread responsible for executing scripts dispatched by the DeviceThread.

    Workers fetch tasks from a shared work pool and process them using the
    device's `process_work` method. They signal when their current task is done.
    """

    def __init__(self, worker_id, work_pool, work_available, work_pool_empty, work_lock, device):
        """
        @brief Initializes a Worker thread.

        @param worker_id: A unique identifier for this worker.
        @param work_pool: The shared list of scripts awaiting execution.
        @param work_available: A Semaphore to signal when work is available in the pool.
        @param work_pool_empty: An Event to signal when the work pool becomes empty.
        @param work_lock: A Lock to protect concurrent access to the 'work_pool'.
        @param device: The parent Device object, used for accessing its processing methods.
        """
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.work_pool = work_pool
        self.work_available = work_available
        self.work_pool_empty = work_pool_empty
        self.work_lock = work_lock
        self.device = device
        self.work_done = Event() # Event to signal when the worker has completed its current task.
        self.work_done.set() # Initially, the worker is considered done (no work yet).

    def run(self):
        """
        @brief The main execution loop for the Worker thread.

        Workers continuously wait for work to become available in the shared
        work pool. Once work is available, they acquire it, process it,
        and then signal their completion before looking for more work.
        """
        # Block Logic: The worker's main loop for continuous work processing.
        while True:
            # Wait until work is available (semaphore is released by DeviceThread).
            self.work_available.acquire()
            self.work_lock.acquire() # Acquire lock to safely access the work pool.
            self.work_done.clear() # Clear work_done event to indicate worker is busy.

            # Block Logic: Check if the work pool is empty after acquiring the lock.
            # This handles the case where the DeviceThread might signal termination.
            if not self.work_pool:
                self.work_lock.release()
                return # If the pool is empty, and no more work is expected, terminate.

            # Get the next script to process from the work pool.
            script, location, neighbours = self.work_pool.pop(0)

            # Block Logic: If the work pool becomes empty after popping an item, set the event.
            if not self.work_pool:
                self.work_pool_empty.set() # Signal that the work pool is now empty.

            self.work_lock.release() # Release the work pool lock.

            # Process the script using the device's processing method.
            self.device.process_work(script, location, neighbours)

            self.work_done.set() # Set work_done event to signal completion of the task.
