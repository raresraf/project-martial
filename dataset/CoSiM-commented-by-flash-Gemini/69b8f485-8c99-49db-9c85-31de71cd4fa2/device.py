"""
@69b8f485-8c99-49db-9c85-31de71cd4fa2/device.py
@brief This module implements a distributed simulation or data processing system
       with a master-worker pattern within each device.

It defines four core classes:
- `ReusableBarrier`: A custom barrier synchronization mechanism for threads that is reusable.
- `Device`: Represents a computational node that manages its sensor data,
  worker threads, and coordinates with a supervisor and other devices. It also handles
  the distribution of scripts to its worker threads.
- `WorkerThread`: Worker threads spawned by a `Device` to execute assigned scripts
  at specific locations, handling data access and propagation.
- `DeviceThread`: The main thread for a `Device`, responsible for fetching neighbor
  information, synchronizing timepoints, and distributing scripts to `WorkerThread`s.

The system relies on `threading` primitives (Lock, Event, Thread, Semaphore)
and `Queue` for concurrency, synchronization, and task distribution, allowing
parallel processing of scripts across multiple devices and within a single device.

Algorithm:
- Decentralized processing: Each `Device` operates semi-autonomously.
- Master-Worker pattern: `DeviceThread` acts as master, `WorkerThread`s as workers.
- Timepoint synchronization: Devices (via `DeviceThread`) and their workers synchronize
  at discrete timepoints using a custom reusable barrier.
- Asynchronous script execution: `WorkerThread`s pull scripts from a `Queue` for execution.
- Distributed locking: Location-specific locks ensure data consistency across devices
  when `WorkerThread`s modify shared data.
- Data gathering and propagation: `WorkerThread`s gather data from neighbors and propagate
  results after script execution.

Time Complexity:
- `ReusableBarrier.__init__`: O(1)
- `ReusableBarrier.wait`: O(num_threads) in worst case (semaphore releases).
- `Device.__init__`: O(num_workers) for starting worker threads.
- `Device.setup_devices`: O(D + L) where D is number of devices, L is total unique locations.
- `WorkerThread.run`: O(S * N_neighbors * L_locations) per timepoint, where S is number of scripts processed,
                       N_neighbors is number of neighbors, L_locations is average unique locations accessed.
- `DeviceThread.run`: O(T * (S + num_workers)) where T is timepoints, S is number of scripts.

Space Complexity:
- `ReusableBarrier`: O(1)
- `Device`: O(L) for locks per location, O(num_workers) for threads, O(S_max) for scripts.
- `WorkerThread`: O(1) beyond script and data storage.
- `DeviceThread`: O(N_neighbors) for storing neighbors, O(S_max) for scripts.
"""

from Queue import Queue
from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads using semaphores.
           Threads wait in two phases, ensuring all threads complete a phase before proceeding
           and allowing the barrier to be used multiple times.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that must reach the barrier
                            in each phase before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Functional Utility: Counter for the first phase of the barrier.
        self.count_threads2 = [self.num_threads] # Functional Utility: Counter for the second phase of the barrier.
        self.count_lock = Lock()                 # Functional Utility: Protects access to the thread counters.
        self.threads_sem1 = Semaphore(0)         # Functional Utility: Semaphore for the first phase, initialized to 0 (all blocked).
        self.threads_sem2 = Semaphore(0)         # Functional Utility: Semaphore for the second phase, initialized to 0 (all blocked).

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all other
               participating threads have also called `wait()`. This involves two
               phases of synchronization to ensure reusability.
        Pre-condition: `num_threads` is properly set, and threads are ready to synchronize.
        Post-condition: All threads have completed both phases of the barrier and are released.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single phase of the barrier synchronization.
        @param count_threads: The counter list for the current phase.
        @param threads_sem: The semaphore for the current phase.
        Pre-condition: `count_threads` reflects the number of threads remaining for this phase.
        Invariant: `count_threads[0]` accurately tracks threads arriving; `threads_sem` controls release.
        Post-condition: All threads have passed this phase, and the counter is reset.
        """
        # Block Logic: Atomically decrements the counter and manages semaphore releases.
        with self.count_lock: # Functional Utility: Acquires a lock to protect the shared counter.
            count_threads[0] -= 1
            # Block Logic: If this is the last thread in the phase, release all waiting threads.
            # Pre-condition: `count_threads[0]` is 0.
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release() # Functional Utility: Releases all threads waiting on this semaphore.
                count_threads[0] = self.num_threads # Functional Utility: Resets the counter for the next use.
        threads_sem.acquire() # Functional Utility: Waits (blocks) until released by the last thread.


class Device(object):
    """
    @brief Represents a computational device in a distributed simulation.
           It manages worker threads, processes scripts, and coordinates data
           with a supervisor and neighboring devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing initial sensor data for various locations.
        @param supervisor: The central supervisor object for coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []               # Functional Utility: List to store scripts assigned to this device.
        self.timepoint_done = Event()   # Functional Utility: Event to signal end of timepoint script assignment.

        self.barrier = None             # Functional Utility: Reference to the global ReusableBarrier.
        self.locks = {}                 # Functional Utility: Dictionary of locks, keyed by location, for data access.
        self.queue = Queue()            # Functional Utility: Queue for distributing scripts to worker threads.
        self.workers = [WorkerThread(self) for _ in range(8)] # Functional Utility: List of worker threads for script execution.

        # Functional Utility: Initializes and starts the main device thread.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Functional Utility: Starts all worker threads.
        for thread in self.workers:
            thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared synchronization primitives (barrier and locks)
               across all devices in the simulation. This setup is typically
               performed by a single designated device (device_id == 0).
        @param devices: A list of all Device instances in the simulation.
        Pre-condition: All Device instances have been initialized.
        Post-condition: All devices share the same barrier and location-specific locks.
        """
        # Block Logic: Ensures that global setup is performed only once by device with ID 0.
        if self.device_id == 0:
            # Functional Utility: Creates a global reusable barrier for all devices (used by DeviceThread).
            barrier = ReusableBarrier(len(devices))

            locks = {} # Invariant: 'locks' will store a unique lock for each unique location across all devices.

            # Block Logic: Iterates through all devices and their sensor data to identify all unique locations,
            # then creates a lock for each unique location to manage concurrent access.
            for device in devices:
                for location in device.sensor_data:
                    if not location in locks:
                        locks[location] = Lock()

            # Block Logic: Assigns the globally created barrier and location-specific locks to all devices.
            # Pre-condition: 'barrier' and 'locks' are initialized.
            # Post-condition: Each device holds references to the shared synchronization objects.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be processed by a worker thread at a specific location.
               If `script` is None, it signals the end of script assignment for the timepoint.
        @param script: The script object to execute.
        @param location: The location ID where the script should be executed.
        Pre-condition: The DeviceThread is waiting for scripts or timepoint completion.
        Post-condition: `script` is added to `self.scripts`, or `timepoint_done` event is set.
        """
        # Block Logic: Appends the script and its location to the list of pending scripts.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: Signals to the main DeviceThread that all scripts
            # for the current timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.
        @param location: The location ID for which to retrieve data.
        @return: The sensor data for the given location, or None if not present.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specified location.
        @param location: The location ID for which to update data.
        @param data: The new data value to set for the location.
        Pre-condition: `location` exists in `self.sensor_data`.
        Post-condition: `self.sensor_data[location]` is updated with `data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the main DeviceThread and subsequently all WorkerThreads.
        Pre-condition: Device is operational.
        Post-condition: All threads associated with this device have terminated.
        """
        self.thread.join() # Functional Utility: Waits for the main DeviceThread to complete its execution.


class WorkerThread(Thread):
    """
    @brief A worker thread managed by a Device. It continuously fetches scripts from
           a queue, executes them, and manages data access to shared locations.
    """

    def __init__(self, device):
        """
        @brief Initializes a WorkerThread instance.
        @param device: The parent Device instance that spawned this thread.
        """
        Thread.__init__(self) # Functional Utility: Initializes the base Thread class.
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the WorkerThread.
               Continuously retrieves script-location pairs from the device's queue,
               acquires location-specific locks, gathers data, executes the script,
               and propagates results to neighboring devices and its own device.
        Pre-condition: The device's queue may contain script tasks.
        Invariant: The thread continues to process items from the queue until a None
                   item (shutdown signal) is encountered.
        Post-condition: The thread terminates after processing all tasks and receiving
                        the shutdown signal.
        """
        # Block Logic: Main loop for continuously processing scripts from the queue.
        while True:
            item = self.device.queue.get() # Functional Utility: Retrieves a task (script, location) from the queue. Blocks if queue is empty.
            # Block Logic: Checks for a shutdown signal from the main DeviceThread.
            # Pre-condition: 'item' is retrieved from the queue.
            # Post-condition: If 'item' is None, the thread breaks from the loop and terminates.
            if item is None:
                break

            (script, location) = item # Functional Utility: Unpacks the script and location from the queue item.

            # Block Logic: Acquires a lock for the specific location to ensure exclusive access to its data.
            # Invariant: Data at 'location' is protected from concurrent modification during script execution.
            with self.device.locks[location]:
                script_data = [] # Invariant: 'script_data' will accumulate relevant data for the script.

                # Block Logic: Gathers sensor data for the current 'location' from neighboring devices.
                # Pre-condition: `self.device.neighbours` contains references to neighboring devices.
                # Invariant: `script_data` grows with valid data from neighbors.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Block Logic: Gathers sensor data for the current 'location' from its own device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if any data was gathered and updates relevant device data.
                # Pre-condition: `script_data` is not empty.
                # Post-condition: The script's result is propagated to neighbors and the device itself.
                if script_data != []:
                    result = script.run(script_data) # Functional Utility: Executes the assigned script with collected data.

                    # Block Logic: Propagates the script's result to neighboring devices.
                    # Invariant: Each neighbor device's sensor data at 'location' is updated with 'result'.
                    for device in self.device.neighbours:
                        device.set_data(location, result)

                    # Functional Utility: Updates the current device's own sensor data at 'location'.
                    self.device.set_data(location, result)

            self.device.queue.task_done() # Functional Utility: Signals that the task retrieved from the queue has been completed.


class DeviceThread(Thread):
    """
    @brief The main thread for a Device. It handles fetching neighbor information,
           synchronizing timepoints, and orchestrating script distribution to WorkerThreads.
    """

    def __init__(self, device):
        """
        @brief Initializes a DeviceThread instance.
        @param device: The parent Device instance that this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Functional Utility: Initializes base Thread with a descriptive name.
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
               It fetches neighbor data, waits for timepoint synchronization,
               puts all assigned scripts into the work queue for WorkerThreads,
               waits for all scripts to complete, and then synchronizes with
               other DeviceThreads via a global barrier.
        Pre-condition: The device is initialized and ready to operate.
        Invariant: The thread continues to loop, processing timepoints, until
                   `self.device.neighbours` becomes None (shutdown signal).
        Post-condition: The thread terminates after processing all timepoints
                        and receiving the shutdown signal.
        """
        # Block Logic: Main loop for processing timepoints.
        while True:
            # Functional Utility: Fetches the list of neighboring devices from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Checks if a shutdown signal (None neighbors) has been received.
            # Pre-condition: `self.device.neighbours` is updated.
            # Post-condition: If `self.device.neighbours` is None, the thread breaks and terminates.
            if self.device.neighbours is None:
                break

            # Functional Utility: Waits until all scripts for the current timepoint have been assigned
            # by the Device's `assign_script` method.
            self.device.timepoint_done.wait()

            # Functional Utility: Clears the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()

            # Block Logic: Places all scripts assigned to this device into the work queue
            # for processing by WorkerThreads.
            # Invariant: Each (script, location) pair from `self.device.scripts` is added to the queue.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location))

            # Functional Utility: Blocks until all tasks (scripts) currently in the queue
            # have been retrieved and marked as done by WorkerThreads.
            self.device.queue.join()

            # Functional Utility: Synchronizes with other DeviceThreads via the global barrier,
            # ensuring all devices complete their timepoint processing before proceeding.
            self.device.barrier.wait()

        # Block Logic: After receiving shutdown signal, sends None signals to worker queues
        # to gracefully shut down all WorkerThreads.
        for _ in range(8): # Functional Utility: Sends a None item to the queue for each worker thread as a shutdown signal.
            self.device.queue.put(None)

        # Functional Utility: Waits for all WorkerThreads to finish their current tasks and terminate.
        for thread in self.device.workers:
            thread.join()
