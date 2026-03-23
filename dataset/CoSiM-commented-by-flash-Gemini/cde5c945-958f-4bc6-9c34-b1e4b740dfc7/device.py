"""
@file device.py
@brief Implements a distributed device simulation framework where each device manages
       its own pool of worker threads (`DeviceWorker`) to process scripts.
       This variant features a central task queue per device, coordinated lock acquisition
       for data locations, and a global conditional barrier for inter-device synchronization.

Algorithm:
- **Centralized Synchronization Setup:** Device 0 initializes a global `acquire_stage_lock`
  (for coordinating access to `location_data_lock` during acquisition) and a
  `ReusableBarrierCond` for all devices. Other devices wait for Device 0 to finish
  initialization before linking to these shared resources.
- **Per-Location Reentrant Locks:** Each `Device` creates a dictionary `location_data_lock`
  containing `RLock` instances for each of its sensor data locations. `RLock` allows
  a single thread to acquire the same lock multiple times.
- **DeviceThread Lifecycle:**
    1. The `DeviceThread` starts all its `DeviceWorker` threads.
    2. It continuously fetches `neighbours` from the supervisor. If `None`, it signals
       its `DeviceWorker`s to stop and breaks.
    3. It waits on a `Condition` variable (`script_condition`) until `timepoint_ended`
       is `True` (signaling all scripts for the current timepoint have been assigned).
    4. It puts all assigned `scripts` into its `work_queue`.
    5. It then blocks until all tasks in `work_queue` are marked as done (`work_queue.join()`).
    6. It resets `timepoint_ended` and notifies `script_condition` for the next cycle.
    7. Finally, it synchronizes with all other devices at the global `sync_barrier`.
- **DeviceWorker Execution:**
    1. `DeviceWorker`s continuously fetch tasks (scripts) from their device's `work_queue`.
    2. **Coordinated Lock Acquisition:** To avoid deadlocks and ensure consistent data access,
       a `DeviceWorker` first acquires the global `acquire_stage_lock`. Then, it attempts
       to acquire the `RLock` for the relevant data `location` for its own device AND for
       any neighbor devices also operating on that location. This sequence ensures all
       necessary locks are held before data access.
    3. It gathers `script_data` from its `self_device` and `acquired_devices`.
    4. It executes the `script.run()` method.
    5. It updates data on all `acquired_devices`.
    6. It releases all `location_data_lock`s that it acquired, then releases `acquire_stage_lock`.

Time Complexity:
- This framework involves significant overhead due to multiple layers of threading,
  inter-thread communication (queues, events, conditions), and coordinated lock
  acquisition. The complexity is highly dependent on:
    - The number of devices (N).
    - The number and complexity of scripts (S) per device.
    - The network topology (number of neighbors per device).
    - Contention for `acquire_stage_lock` and `location_data_lock`s.
- Global synchronization points (`sync_barrier.wait()`) and queue operations can be bottlenecks.

Space Complexity:
- O(N_devices * (D + S)) for storing device-specific sensor data (D) and scripts (S).
- Additional overhead for:
    - `location_data_lock` (dictionary of `RLock`s, grows with unique data locations).
    - `ReusableBarrierCond`, `Queue`, `Event`, `Condition`, `Lock`, `RLock` internal structures.
    - Thread stacks for `DeviceThread`s and `DeviceWorker`s.
"""

from threading import Event, Thread, Lock, Condition, RLock # Import necessary threading primitives.
from Queue import Queue                                  # Import Queue for task distribution.
from barrier import ReusableBarrierCond                  # Import the custom ReusableBarrierCond class.


class Device(object):
    """
    @class Device
    @brief Represents a single simulated device in the distributed system.
           Manages its local sensor data, assigned scripts, and its worker pool.
           It interacts with a central supervisor and participates in global synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id: Unique identifier for this device.
        @param sensor_data: A dictionary containing sensor readings/data specific to this device.
        @param supervisor: The central supervisor object responsible for managing all devices.
        """
        self.sync_barrier = None       # Reference to the global ReusableBarrierCond.
        self.acquire_stage_lock = None # Reference to the global Lock for coordinating lock acquisition.
        self.device_init_event = Event() # Event to signal that this device's setup is complete.

        # Dictionary of Reentrant Locks (RLock), one for each data location in sensor_data.
        # RLock allows the same thread to acquire it multiple times.
        self.location_data_lock = {location:RLock() for location in sensor_data}

        self.device_id = device_id     # Unique identifier for the device.
        self.supervisor = supervisor   # Reference to the supervisor object.
        self.sensor_data = sensor_data # Local sensor data (dict: {'location_id': data_value}).

        self.timepoint_ended = False   # Flag to indicate if script assignment for a timepoint has ended.
        self.script_condition = Condition() # Condition variable to coordinate script assignment and processing.
        self.scripts = []              # List to store assigned scripts (script_obj, location_id).

        # Create the main DeviceThread for this device and a pool of DeviceWorkers.
        self.thread = DeviceThread(0, self) # The '0' for thread_id might be a placeholder.
        self.worker_pool = [DeviceWorker(i, self) for i in xrange(1, 9)] # 8 worker threads per device.

        self.neighbours = []           # List of neighboring Device instances.
        self.work_queue = Queue()      # Queue for distributing tasks (scripts) to DeviceWorkers.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives among all devices.
               Device 0 initializes global resources; other devices wait for it.
        @param devices: A list of all Device instances in the simulation.
        Functional Utility: Coordinates the initialization of global threading and
                            synchronization components to ensure consistent behavior across devices.
        """
        # Block Logic: Device 0 acts as the coordinator to initialize global shared resources.
        if self.device_id == 0:
            self.acquire_stage_lock = Lock() # Global lock to coordinate location lock acquisitions.
            self.sync_barrier = ReusableBarrierCond(len(devices)) # Initialize the global conditional barrier.
        else:
            # Block Logic: Other devices wait for Device 0 to complete its initialization.
            for device in devices:
                if device.device_id == 0:
                    device.device_init_event.wait() # Blocks until Device 0 signals it's initialized.
                    # Link to the shared global resources initialized by Device 0.
                    self.sync_barrier = device.sync_barrier
                    self.acquire_stage_lock = device.acquire_stage_lock
        self.device_init_event.set() # Signal that this device is initialized.
        self.thread.start()          # Start the main DeviceThread.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific data location.
               Uses a condition variable for thread-safe script assignment and signaling.
        @param script: The script object to be executed.
        @param location: The data location the script operates on.
        Functional Utility: Queues scripts for processing by the device's worker pool.
        """
        # Block Logic: Protects access to `scripts` list and `timepoint_ended` flag using a Condition variable.
        with self.script_condition:
            # Block Logic: Waits if the current timepoint has ended but scripts are still being assigned.
            while self.timepoint_ended:
                self.script_condition.wait() # Release lock and wait for notification.
            if script is not None:
                self.scripts.append((script, location)) # Add the script and its location.
            else:
                self.timepoint_ended = True # Mark that script assignment for this timepoint has ended.
                self.script_condition.notify_all() # Notify the DeviceThread to start processing.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
               Access is protected by the RLock associated with the location.
        @param location: The key corresponding to the data location.
        @return The data at the specified location, or None if the location does not exist.
        Functional Utility: Provides thread-safe access to the device's local sensor data.
        """
        if location in self.sensor_data:
            with self.location_data_lock[location]: # Acquire RLock for this specific location.
                return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a specific location.
               Access is protected by the RLock associated with the location.
        @param location: The key corresponding to the data location.
        @param data: The new data value to set.
        Functional Utility: Provides thread-safe modification of the device's local sensor data.
        """
        if location in self.sensor_data:
            with self.location_data_lock[location]: # Acquire RLock for this specific location.
                self.sensor_data[location] = data

    def acquire_location(self, location):
        """
        @brief Attempts to acquire the RLock for a specific data location.
        @param location: The data location for which to acquire the lock.
        @return True if the lock was acquired, False otherwise.
        Functional Utility: Used by workers to coordinate access to data.
        """
        if location in self.location_data_lock:
            self.location_data_lock[location].acquire() # Acquire the RLock.
            return True
        return False

    def release_location(self, location):
        """
        @brief Releases the RLock for a specific data location.
        @param location: The data location for which to release the lock.
        Functional Utility: Used by workers to release coordinated data access.
        """
        if location in self.location_data_lock:
            try:
                self.location_data_lock[location].release() # Release the RLock.
            except RuntimeError:
                # Handle cases where the lock might not have been acquired by the current thread (e.g., error).
                pass

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's main thread and worker pool.
        Functional Utility: Ensures proper termination and cleanup of all associated threads.
        """
        self.thread.join() # Waits for the DeviceThread to complete its execution.


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief The main control thread for a `Device`. It manages the lifecycle of scripts,
           distributes them to `DeviceWorker`s via a queue, and handles global synchronization.
    """

    def __init__(self, thread_id, device):
        """
        @brief Initializes the DeviceThread.
        @param thread_id: A unique identifier for this thread (0 for main DeviceThread).
        @param device: The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % thread_id)
        self.thread_id = thread_id # Identifier for the thread.
        self.device = device       # Reference to the associated Device instance.

    def stop_device(self):
        """
        @brief Signals all `DeviceWorker` threads in the pool to stop and waits for them to terminate.
        Functional Utility: Gracefully shuts down the worker pool when the simulation ends.
        """
        # Block Logic: Puts a `None` sentinel value into the work queue for each worker.
        for _ in xrange(len(self.device.worker_pool)):
            self.device.work_queue.put(None)
        self.device.work_queue.join() # Blocks until all items in the queue (including None sentinels) are processed.
        # Block Logic: Joins each worker thread, ensuring they have completed execution.
        for thread in self.device.worker_pool:
            thread.join()

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Continuously fetches neighbors, waits for script assignments,
                            distributes them to `DeviceWorker`s, waits for worker completion,
                            and synchronizes globally.
        """
        # Block Logic: Start all DeviceWorker threads owned by this device.
        for thread in self.device.worker_pool:
            thread.start()

        # Block Logic: The main loop for the device's operational thread.
        # Invariant: The device continuously processes time steps until a shutdown signal is received.
        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours() # Fetches current neighbors.
            if self.device.neighbours is None:
                self.stop_device() # If no neighbors (simulation ended), stop workers.
                break              # Exit the main device loop.

            # Block Logic: Waits until script assignment for the current timepoint has ended.
            with self.device.script_condition:
                while not self.device.timepoint_ended:
                    self.device.script_condition.wait() # Wait until `timepoint_ended` is true.

                # Block Logic: Puts all assigned scripts into the work queue for processing by workers.
                for script in self.device.scripts:
                    self.device.work_queue.put(script)

                self.device.work_queue.join() # Blocks until all scripts in the queue are processed by workers.

                self.device.timepoint_ended = False # Reset flag for the next timepoint.
                self.device.scripts = []            # Clear scripts for the next timepoint.
                self.device.script_condition.notify_all() # Notify any waiting assign_script calls.

            self.device.sync_barrier.wait() # Global synchronization at the conditional barrier.


class DeviceWorker(DeviceThread):
    """
    @class DeviceWorker
    @brief A worker thread managed by a `Device`'s `DeviceThread`.
           It fetches tasks (scripts) from a queue, acquires locks for data locations
           (including neighbors'), executes scripts, and updates data.
    """

    def __init__(self, thread_id, device):
        """
        @brief Initializes a DeviceWorker thread.
        @param thread_id: A unique identifier for this worker thread.
        @param device: The parent `Device` instance this worker belongs to.
        """
        super(DeviceWorker, self).__init__(thread_id, device) # Call parent (DeviceThread) constructor.

    def run(self):
        """
        @brief The main execution loop for the DeviceWorker thread.
        Functional Utility: Continuously picks up tasks, performs coordinated lock acquisition
                            for data locations (including neighbors), executes scripts,
                            updates data, and releases locks.
        """
        # Block Logic: The main loop for the worker. It continuously gets items from the work queue.
        while True:
            item = self.device.work_queue.get() # Get a task (script) from the queue.
            if item is None:
                self.device.work_queue.task_done() # Mark task as done (for the None sentinel).
                break                              # Exit loop if sentinel received.
            (script, location) = item # Unpack the script and its data location.

            acquired_devices = [] # List to keep track of devices whose locks were successfully acquired.
            script_data = []      # List to aggregate data for the script.

            # Block Logic: Coordinated lock acquisition using the global `acquire_stage_lock`.
            # Functional Utility: Prevents deadlocks by serializing the acquisition of multiple
            #                     `location_data_lock`s across devices.
            with self.device.acquire_stage_lock:
                # Attempt to acquire lock for its own device's data location.
                if self.device.acquire_location(location):
                    acquired_devices.append(self.device)
                
                # Attempt to acquire locks for neighbor devices' data locations.
                for device in self.device.neighbours:
                    if device.device_id != self.device.device_id: # Avoid acquiring lock on self again.
                        if device.acquire_location(location):
                            acquired_devices.append(device)

            # Block Logic: Gather data from all devices whose locks were acquired.
            for device in acquired_devices:
                data = device.get_data(location) # Retrieves data from the device (protected by its RLock).
                if data is not None:
                    script_data.append(data)

            # Block Logic: If data was gathered, execute the script and update data.
            if len(script_data) > 0:
                result = script.run(script_data) # Execute the script.

                # Update data on all devices whose locks were acquired.
                for device in acquired_devices:
                    device.set_data(location, result)

            # Block Logic: Release all locks that were acquired in this task.
            for device in acquired_devices:
                device.release_location(location)

            self.device.work_queue.task_done() # Mark this task as done in the queue.
