"""
@dced99a1-6c3b-470b-88e2-907adbb24afd/device.py
@brief Defines core components for simulating a distributed sensor network or device system.
This module provides classes for devices, their operational threads, and a reusable
barrier synchronization mechanism, enabling simulation of concurrent operations
and data exchange across multiple simulated entities. This version introduces
a `Queue`-based worker pool (`WorkerThread`) for script execution, where scripts
process aggregated data and update sensor values based on a conditional logic (e.g., maximum propagation).

Domain: Distributed Systems, Concurrency, Simulation, Worker Pool.
"""

from Queue import Queue
from threading import Semaphore, Lock
from threading import Event, Thread


class Device(object):
    """
    @brief Represents a single device in the simulated distributed system.
    Each device manages its own sensor data, processes assigned scripts
    via a worker pool, and interacts with a supervisor to coordinate
    with other devices.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data readings
                            keyed by location (e.g., "temperature": 25.5).
        @param supervisor: A reference to the supervisor object that
                           manages inter-device communication and coordination.
        """
        self.device_id = device_id
        # Stores the sensor data for this device.
        self.read_data = sensor_data
        self.supervisor = supervisor
        # A queue to hold scripts and their locations for processing by worker threads.
        self.active_queue = Queue()
        # A temporary list to store assigned scripts before they are dispatched to the queue.
        self.scripts = []
        # The main thread responsible for managing this device's operations and worker threads.
        self.thread = DeviceThread(self)
        # Represents the current simulation timepoint or round.
        self.time = 0
        # Placeholder for the shared barrier object, initialized by device 0.
        self.new_round = None
        # Placeholder for the list of all devices in the simulation, initialized by device 0.
        self.devices = None


    def __str__(self):
        """
        @brief Returns a string representation of the device.
        @return A string in the format "Device {device_id}".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared resources (barrier) across devices and starts the main thread.
        This method ensures that all devices share the same synchronization barrier.
        The first device (device_id 0) is responsible for initializing the barrier
        and sharing it with all other devices. The main device thread is started here.
        @param devices: A list of all Device objects in the simulation.
        """
        # Block Logic: Only Device 0 initializes the shared barrier and the global device list.
        if self.device_id == 0:
            # Reusable barrier for synchronizing all DeviceThreads at the end of each round.
            self.new_round = ReusableBarrierSem(len(devices))
            # Store a reference to all devices for access by worker threads.
            self.devices = devices
            # Propagate the initialized barrier to all other devices.
            for device in self.devices:
                device.new_round = self.new_round
                device.devices = self.devices # Also propagate the device list

        # Start the main operational thread for this device.
        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device or dispatches queued scripts to workers.
        If a script is provided, it's added to a temporary list. If `script` is `None`,
        it signals the end of script assignments for the current round, and all
        pending scripts are moved from the temporary list to the `active_queue`
        for processing by worker threads. Sentinel values are also added to stop workers.
        @param script: The script object to be executed, or None to signal dispatch.
        @param location: The data location pertinent to the script's execution.
        """
        # Block Logic: Handles either adding a script to a pending list or dispatching all pending scripts.
        if script is not None:
            # If a script is provided, add it to the list for the current round.
            self.scripts.append((script, location))
        else:
            # If script is None, it's time to dispatch all collected scripts to the active queue.
            for (script_item, loc_item) in self.scripts:
                self.active_queue.put((script_item, loc_item))
            # Add sentinel values to the queue to signal worker threads to terminate after processing.
            # Assuming 8 worker threads per device.
            for x in range(8):
                self.active_queue.put((-1, -1))
            # Clear the temporary scripts list for the next round.
            self.scripts = []


    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.
        @param location: The location for which to retrieve data.
        @return The sensor data at the given location, or None if the location
                does not exist in the device's `read_data`.
        """
        # Inline: Safely retrieve data using dictionary's get method to handle missing locations.
        return self.read_data[location] if location in self.read_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specified location.
        @param location: The location whose data needs to be updated.
        @param data: The new data value to set for the location.
        """
        # Block Logic: Updates sensor data only if the location already exists.
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread.
        Ensures proper termination by waiting for the device's thread to complete its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the execution lifecycle of a single Device.
    This thread is responsible for coordinating with the supervisor to get
    neighbor information, managing a pool of `WorkerThread`s for parallel
    script execution, and synchronizing with other `DeviceThread`s using a barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread for a given device.
        @param device: The Device object that this thread will manage.
        """
        # Functional Utility: Initializes the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # The number of worker threads to be spawned for parallel script execution.
        self.workers_number = 8

    def run(self):
        """
        @brief The main execution loop for the device thread.
        This loop continuously retrieves neighbor information, launches
        worker threads to process scripts, waits for their completion,
        and synchronizes with other device threads using a barrier.
        """
        # Block Logic: Retrieves initial neighbors from the supervisor.
        neighbours = self.device.supervisor.get_neighbours()
        # Block Logic: The main simulation loop, continuing until no neighbors are found
        # (indicating simulation termination or a paused state).
        while True:
            # Functional Utility: Resets the list of workers for the current round.
            self.workers = []
            # Store the current neighbors list in the device for access by worker threads.
            self.device.neighbours = neighbours
            # Pre-condition: If no neighbors are returned, the simulation for this device ends.
            if neighbours is None:
                break

            # Block Logic: Spawns and starts `workers_number` of `WorkerThread` instances.
            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            # Block Logic: Waits for all worker threads to complete their tasks for the current round.
            for worker in self.workers:
                worker.join()
            # Block Logic: Synchronizes with all other DeviceThreads using the shared barrier.
            # Invariant: All DeviceThreads must reach this point before any can proceed to the next round.
            self.device.new_round.wait()
            # Block Logic: Fetches the neighbors for the next simulation round.
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):
    """
    @brief A worker thread responsible for processing a single script from the device's active queue.
    These threads aggregate data from the device and its neighbors, execute
    the assigned script, and conditionally update sensor data based on the script's result.
    """

    def __init__(self, device):
        """
        @brief Initializes a WorkerThread for a given device.
        @param device: The parent Device object from which to get tasks and data.
        """
        # Functional Utility: Initializes the base Thread class with a descriptive name.
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution logic for a worker thread.
        Continuously pulls scripts from the `active_queue`, processes them
        by aggregating data, running the script, and applying conditional updates.
        """
        # Block Logic: The continuous loop for processing scripts from the active queue.
        while True:
            # Functional Utility: Retrieves a script and its location from the active queue.
            # This operation blocks until an item is available.
            script, location = self.device.active_queue.get()
            # Pre-condition: If a sentinel value (-1, -1) is received, the worker terminates.
            if script == -1:
                break
            script_data = []
            # List to keep track of devices that contributed data for the current script.
            matches = []
            
            # Block Logic: Collects sensor data from neighboring devices at the specified location.
            for device in self.device.neighbours:
                data = device.get_data(location)
                # Invariant: Only valid (non-None) data is collected, and the contributing device is noted.
                if data is not None:
                    matches.append(device)
                    script_data.append(data)
            # Block Logic: Collects sensor data from the current device itself.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
                matches.append(self.device)

            # Block Logic: Executes the script only if enough data (more than one source) has been collected.
            # This condition ensures data aggregation has occurred.
            if len(script_data) > 1:
                # Functional Utility: Executes the assigned script with the aggregated data.
                result = script.run(script_data)
                # Block Logic: Conditionally updates the sensor data on matching devices.
                # Optimization: This loop updates the data on all contributing devices if the
                # script's result is "better" (e.g., greater than) the existing value.
                for device in matches:
                    old_value = device.get_data(location)
                    # Pre-condition: Data is updated only if the script's result is higher than the old value.
                    if old_value is not None and old_value < result:
                        device.set_data(location, result)


class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier synchronization mechanism using semaphores.
    This barrier allows a fixed number of threads to synchronize multiple times,
    ensuring that no thread proceeds past the barrier until all participating
    threads have reached it. It uses a two-phase approach to allow reusability.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that must reach the
                            barrier for it to be lifted.
        """
        self.num_threads = num_threads
        # Counter for the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # Counter for the second phase of the barrier, enabling reusability.
        self.count_threads2 = self.num_threads
        # Lock to protect access to the thread counters.
        self.counter_lock = Lock()
        # Semaphore for the first synchronization phase. Initialized to 0
        # so threads wait until all have arrived.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second synchronization phase. Initialized to 0
        # for reusability, ensures threads wait for reset.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this point.
        Orchestrates the two-phase barrier synchronization.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief The first phase of the reusable barrier.
        Threads decrement a counter and the last thread to reach zero
        releases all waiting threads via a semaphore, then resets the counter.
        """
        # Block Logic: Critical section for safely decrementing the thread counter.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Invariant: If this is the last thread, release all waiting threads.
            if self.count_threads1 == 0:
                # Release all threads waiting on threads_sem1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the counter for the next use of the barrier.
                self.count_threads1 = self.num_threads

        # Wait until all other threads have reached phase 1 and the semaphore is released.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief The second phase of the reusable barrier.
        Similar to phase 1, but uses a different semaphore to allow
        the barrier to be reused.
        """
        # Block Logic: Critical section for safely decrementing the thread counter for phase 2.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Invariant: If this is the last thread, release all waiting threads.
            if self.count_threads2 == 0:
                # Release all threads waiting on threads_sem2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the counter for the next use of the barrier.
                self.count_threads2 = self.num_threads

        # Wait until all other threads have reached phase 2 and the semaphore is released.
        self.threads_sem2.acquire()
