"""
@6a0b8942-cb4d-428b-9f6b-966723eaec7c/device.py
@brief This module implements a distributed simulation or data processing system
       with a master-worker pattern within each device, utilizing a custom
       reusable barrier for synchronization.

It defines four core classes:
- `Device`: Represents a computational node managing sensor data, scripts,
  and orchestrating its worker threads.
- `DeviceThread`: The main thread for a `Device`, responsible for timepoint
  advancement, fetching neighbor data, and managing worker threads' lifecycle.
- `WorkerThread`: Worker threads spawned by a `DeviceThread` to execute assigned
  scripts for specific locations, handling data retrieval and updates.
- `ReusableBarrierSem`: A custom two-phase reusable barrier synchronization
  mechanism utilizing semaphores.

The system relies on `threading` primitives (Lock, Event, Thread, Semaphore)
and `Queue` for concurrency, synchronization, and task distribution, allowing
parallel processing of scripts across multiple devices and within a single device.

Algorithm:
- Decentralized processing: Each `Device` operates semi-autonomously.
- Master-Worker pattern: `DeviceThread` acts as master, creating and managing `WorkerThread`s.
- Timepoint synchronization: `DeviceThread`s synchronize at discrete timepoints using `ReusableBarrierSem`.
- Asynchronous script execution: `WorkerThread`s pull script-location pairs from a `Queue`.
- Data gathering and conditional propagation: `WorkerThread`s gather data from neighbors and
  itself. If the script result is greater than the existing data at a location,
  it updates the data. (This implies a specific type of problem, e.g., finding maximums).

Time Complexity:
- `Device.__init__`: O(1)
- `Device.setup_devices`: O(D) where D is the total number of devices (only executed by device_id 0).
- `DeviceThread.run`: O(T * (W + S + N_neighbors)) where T is number of timepoints,
                      W is number of workers (threads created/joined per timepoint),
                      S is number of scripts, N_neighbors is the cost of getting neighbors.
- `WorkerThread.run`: O(S_per_worker * (N_neighbors + 1)) where S_per_worker is scripts per worker,
                      N_neighbors is number of neighbors.
- `ReusableBarrierSem.__init__`: O(1)
- `ReusableBarrierSem.wait`: O(num_threads) due to semaphore releases.

Space Complexity:
- `Device`: O(S_max) for scripts, O(W) for workers, O(Q_max) for queue.
- `DeviceThread`: O(N_neighbors) for neighbors.
- `WorkerThread`: O(1) apart from script_data.
- `ReusableBarrierSem`: O(1)
"""

from Queue import Queue
from threading import Semaphore, Lock
from threading import Event, Thread


class Device(object):
    """
    @brief Represents a computational node in the distributed simulation.
           Manages sensor data, a queue for scripts, and its associated threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique integer identifier for this device.
        @param sensor_data: A dictionary containing initial sensor data, keyed by location.
        @param supervisor: The central supervisor object responsible for orchestrating devices.
        """
        self.device_id = device_id
        self.read_data = sensor_data        # Functional Utility: Stores sensor data for various locations.
        self.supervisor = supervisor
        self.active_queue = Queue()         # Functional Utility: Queue for distributing scripts to WorkerThreads.
        self.scripts = []                   # Functional Utility: List to store scripts assigned for the current timepoint.
        self.thread = DeviceThread(self)    # Functional Utility: The main thread managing this device's operations.
        self.time = 0                       # Functional Utility: Tracks the current simulation timepoint.
        self.neighbours = None              # Invariant: Stores references to neighboring devices, updated by DeviceThread.
        self.new_round = None               # Functional Utility: Reference to the global ReusableBarrierSem for timepoint synchronization.
        self.devices = None                 # Functional Utility: List of all devices, set by device 0 during setup.


    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared synchronization primitives (ReusableBarrierSem) across all devices.
               This setup is typically performed by a single designated device (device_id == 0).
        @param devices: A list of all Device instances in the simulation.
        Pre-condition: All Device instances have been initialized.
        Post-condition: All devices share the same `ReusableBarrierSem` instance.
        """
        # Block Logic: Ensures that global setup is performed only once by the device with ID 0.
        if self.device_id == 0:
            # Functional Utility: Creates a global reusable barrier for all DeviceThreads.
            self.new_round = ReusableBarrierSem(len(devices))
            self.devices = devices # Functional Utility: Stores reference to all devices.
            # Block Logic: Assigns the globally created barrier to all devices.
            for device in self.devices:
                device.new_round = self.new_round
        # Functional Utility: Starts the main DeviceThread for this device.
        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be processed by a worker thread at a specific location.
               If `script` is None, it signals that all scripts for the current timepoint
               have been assigned, and they can be put into the active queue.
        @param script: The script object to execute.
        @param location: The location ID where the script should be executed.
        Pre-condition: Scripts are being assigned for the current timepoint.
        Post-condition: `script` is added to `self.scripts`, or all collected scripts
                        are put into the `active_queue` and shutdown signals are sent for workers.
        """
        # Block Logic: Appends the script and its location to the list of pending scripts.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: If a None script is received, it signifies the end of script assignment
            # for the timepoint. All collected scripts are put into the queue for workers.
            # Invariant: Each (script, location) pair from `self.scripts` is added to the queue.
            for (script, location) in self.scripts:
                self.active_queue.put((script, location))
            # Block Logic: Sends shutdown signals to all worker threads by placing special
            # (-1, -1) tuples in the queue.
            # Functional Utility: Assumes 8 worker threads per device.
            for x in range(8):
                self.active_queue.put((-1, -1)) # Functional Utility: (-1, -1) acts as a sentinel for worker shutdown.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.
        @param location: The location ID for which to retrieve data.
        @return: The sensor data for the given location, or None if not present.
        """
        return self.read_data[location] if location in self.read_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specified location.
        @param location: The location ID for which to update data.
        @param data: The new data value to set for the location.
        Pre-condition: `location` exists in `self.read_data`.
        Post-condition: `self.read_data[location]` is updated with `data`.
        """
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the main DeviceThread. This will, in turn, manage the shutdown
               of its WorkerThreads.
        Pre-condition: Device is operational.
        Post-condition: The main DeviceThread (and its workers) have terminated.
        """
        self.thread.join() # Functional Utility: Waits for the main DeviceThread to complete its execution.


class DeviceThread(Thread):
    """
    @brief The main thread for a Device. It handles fetching neighbor information,
           creating and joining worker threads for each timepoint, and synchronizing
           with other devices at the end of each round.
    """

    def __init__(self, device):
        """
        @brief Initializes a DeviceThread instance.
        @param device: The parent Device instance that this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Functional Utility: Initializes base Thread with descriptive name.
        self.device = device
        self.workers_number = 8 # Configuration: Number of worker threads this DeviceThread will manage.

    def run(self):
        # Block Logic: Fetches the initial list of neighboring devices from the supervisor.
        # This list is updated at the beginning of each round (timepoint).
        neighbours = self.device.supervisor.get_neighbours()
        # Block Logic: Main loop for processing timepoints (rounds).
        # Invariant: The loop continues as long as valid neighbors are returned by the supervisor.
        while True:
            self.workers = [] # Functional Utility: List to hold worker threads for the current round.
            self.device.neighbours = neighbours # Functional Utility: Updates the device's neighbor list for the current round.
            # Block Logic: Checks if a shutdown signal (None neighbors) has been received.
            # Pre-condition: `neighbours` is updated for the current round.
            # Post-condition: If `neighbours` is None, the thread breaks from the loop and terminates.
            if neighbours is None:
                break

            # Block Logic: Creates and starts a pool of WorkerThreads for the current timepoint.
            # Invariant: `workers_number` of WorkerThreads are created and started.
            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            # Block Logic: Waits for all worker threads to complete their tasks for the current timepoint.
            # Invariant: All WorkerThreads are joined, meaning they have processed all assigned scripts
            # and received their shutdown signals for this batch of work.
            for worker in self.workers:
                worker.join()
            # Functional Utility: Synchronizes with other DeviceThreads via the global barrier,
            # ensuring all devices complete their timepoint processing before advancing.
            self.device.new_round.wait()
            # Block Logic: Fetches updated neighbor information for the next timepoint.
            # Pre-condition: All devices have synchronized at the barrier.
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):
    """
    @brief A worker thread responsible for executing scripts assigned to a Device.
           It retrieves script-location pairs from a queue, gathers data,
           executes the script, and conditionally updates data based on the result.
    """

    def __init__(self, device):
        """
        @brief Initializes a WorkerThread instance.
        @param device: The parent Device instance that this thread controls.
        """
        Thread.__init__(self, name="Worker Thread %d" % device.device_id) # Functional Utility: Initializes base Thread.
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the WorkerThread.
               Continuously retrieves script-location pairs from the device's queue,
               gathers data from its device and neighbors, executes the script,
               and conditionally updates data if the script result is higher.
        Pre-condition: The device's `active_queue` may contain script tasks.
        Invariant: The thread continues to process items from the queue until a
                   sentinel value (-1, -1) is encountered.
        Post-condition: The thread terminates after processing all tasks for the
                        current batch and receiving the shutdown signal.
        """
        # Block Logic: Main loop for continuously processing scripts from the queue.
        while True:
            script, location = self.device.active_queue.get() # Functional Utility: Retrieves a task (script, location) from the queue.
            # Block Logic: Checks for a shutdown signal (sentinel value) from the `assign_script` method.
            # Pre-condition: 'script' and 'location' are retrieved.
            # Post-condition: If 'script' is -1, the thread breaks from the loop and terminates.
            if script == -1:
                break
            script_data = [] # Invariant: 'script_data' will accumulate relevant data for the script.
            matches = []     # Invariant: 'matches' will store devices that provided data for this location.

            # Block Logic: Gathers sensor data for the current 'location' from neighboring devices.
            # Invariant: `script_data` grows with valid data, and `matches` tracks corresponding devices.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    matches.append(device)
                    script_data.append(data)
            # Block Logic: Gathers sensor data for the current 'location' from its own device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
                matches.append(self.device)

            # Block Logic: Executes the script if enough data was gathered and conditionally updates data.
            # Pre-condition: `script_data` contains at least two data points.
            # Post-condition: Data in matched devices is updated only if `result` is greater.
            if len(script_data) > 1: # Functional Utility: Condition implies scripts require multiple data points.
                result = script.run(script_data) # Functional Utility: Executes the assigned script with collected data.
                # Block Logic: Propagates the script's result to matched devices, updating only if the new result is higher.
                # Invariant: Data at 'location' in `device` is updated only if `result` is greater than `old_value`.
                for device in matches:
                    old_value = device.get_data(location)
                    if old_value < result: # Functional Utility: Conditional update, likely for max-finding scenarios.
                        device.set_data(location, result)

            self.device.active_queue.task_done() # Functional Utility: Signals that the task retrieved from the queue has been completed.


class ReusableBarrierSem():
    """
    @brief Implements a two-phase reusable barrier for synchronizing multiple threads.
           It uses semaphores to block and release threads in distinct phases,
           allowing for multiple synchronization points.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that must reach the barrier
                            in each phase before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Functional Utility: Counter for the first phase of the barrier.
        self.count_threads2 = self.num_threads # Functional Utility: Counter for the second phase of the barrier.
        self.counter_lock = Lock()              # Functional Utility: Protects access to the shared counters.
        self.threads_sem1 = Semaphore(0)        # Functional Utility: Semaphore for the first phase, initialized to 0 (all blocked).
        self.threads_sem2 = Semaphore(0)        # Functional Utility: Semaphore for the second phase, initialized to 0 (all blocked).

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all other
               participating threads have completed both phases of the barrier.
        Pre-condition: `num_threads` is correctly set.
        Post-condition: All threads have completed both phases and are released.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Executes the first phase of the barrier synchronization.
        Pre-condition: All threads are entering the first phase.
        Invariant: `count_threads1` accurately tracks threads arriving; `threads_sem1` controls release.
        Post-condition: All threads have passed phase 1, and the counter is reset for this phase.
        """
        # Block Logic: Atomically decrements the counter for phase 1 and manages semaphore releases.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Block Logic: If this is the last thread in phase 1, release all waiting threads.
            if self.count_threads1 == 0:
                # Functional Utility: Releases all threads waiting on 'threads_sem1'.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Functional Utility: Resets the counter for the next use.

        self.threads_sem1.acquire() # Functional Utility: Waits (blocks) until released by the last thread in phase 1.

    def phase2(self):
        """
        @brief Executes the second phase of the barrier synchronization.
        Pre-condition: All threads have successfully completed phase 1.
        Invariant: `count_threads2` accurately tracks threads arriving; `threads_sem2` controls release.
        Post-condition: All threads have passed phase 2, and the counter is reset for this phase.
        """
        # Block Logic: Atomically decrements the counter for phase 2 and manages semaphore releases.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Block Logic: If this is the last thread in phase 2, release all waiting threads.
            if self.count_threads2 == 0:
                # Functional Utility: Releases all threads waiting on 'threads_sem2'.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Functional Utility: Resets the counter for the next use.

        self.threads_sem2.acquire() # Functional Utility: Waits (blocks) until released by the last thread in phase 2.
