


from threading import Event, Thread, Lock, Condition
from reusable_barrier import ReusableBarrier

NUM_THREADS = 8

class Device(object):
    """
    Represents a simulated device within a multi-device system, designed
    for complex synchronization patterns using conditions and events.

    This device manages its sensor data, processes assigned scripts through
    a pool of worker threads, and coordinates tightly with other devices
    using a shared barrier and location-specific synchronization mechanisms
    to ensure data consistency and correct script execution order.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary containing sensor readings,
                                 where keys are locations and values are data.
            supervisor (Supervisor): A reference to the supervisor object
                                     that manages overall simulation and device interactions.
        """
        self.device_id = device_id
        self.supervisor = supervisor

        # Event to signal that initial device setup (like barrier distribution) is complete.
        self.ready_to_start = Event()

        # Lock for protecting access to the sensor_data dictionary.
        self.data_lock = Lock()
        # Dictionary storing this device's sensor data.
        self.sensor_data = sensor_data

        # Dictionary to track if a specific location is currently being processed by a script.
        self.location_busy = {location: False for location in self.sensor_data}
        # Lock for protecting access to the location_busy dictionary.
        self.location_busy_lock = Lock()

        # List to store scripts assigned to this device.
        self.scripts = []
        # Flag indicating if all scripts for the current timestep have been assigned.
        self.scripts_assigned = False
        # Flag indicating if worker threads should start processing scripts.
        self.scripts_enabled = False
        # Index to track how many scripts have been started by workers.
        self.scripts_started_idx = 0
        # Index to track how many scripts have been completed by workers.
        self.scripts_done_idx = 0

        # Lock to protect access to scripts list and associated conditions.
        self.scripts_lock = Lock()
        # Condition variable for workers to wait for new scripts or for scripts_enabled flag.
        self.scripts_condition = Condition(self.scripts_lock)
        # Condition variable for DeviceThread to wait for all scripts to be done.
        self.scripts_done_condition = Condition(self.scripts_lock)

        # Flag to control the main DeviceThread and worker threads' execution.
        self.thread_running = True
        # The main thread for this device's operational logic.
        self.thread = DeviceThread(self)
        # List of worker threads, pre-initialized.
        self.worker_threads = [ScriptWorker(self, i) for i in range(NUM_THREADS)]

        # Start all worker threads upon initialization.
        for thread in self.worker_threads:
            thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures shared synchronization mechanisms for all devices in the system.

        If this device is device_id 0, it initializes the `ReusableBarrier`
        (timestep_barrier) and a dictionary of `location_conditions` for
        fine-grained control over data access. These shared resources are
        then distributed to all other devices. Finally, it signals all devices
        that they are ready to start.

        Args:
            devices (list): A list of all `Device` instances in the system.
        """
        # Only device with device_id 0 is responsible for initializing shared resources.
        if self.device_id == 0:
            # Initialize the shared reusable barrier for synchronizing timesteps.
            timestep_barrier = ReusableBarrier(len(devices))
            # Create a dictionary to hold Condition variables for each location,
            # allowing specific control over data access.
            location_conditions = {}

            # Populate location_conditions with a Condition for every unique location across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if not location in location_conditions:
                        location_conditions[location] = Condition()

            # Distribute the shared location_conditions and timestep_barrier to all devices.
            for device in devices:
                device.location_conditions = location_conditions
                device.timestep_barrier = timestep_barrier

            # Signal all devices that the initial setup is complete and they can proceed.
            for device in devices:
                device.ready_to_start.set()

        # Start the main DeviceThread for this device.
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device.

        If a script is provided, it's added to the device's list of scripts,
        and workers waiting on `scripts_condition` are notified.
        If `script` is None, it signals that script assignment for the current
        timestep is complete, and workers waiting on `scripts_done_condition`
        are notified.

        Args:
            script (object or None): The script object to assign, or None to signal
                                     the end of script assignments for a timestep.
            location (int): The identifier for the location associated with the script.
        """
        # Acquire the scripts_lock to safely modify script-related state.
        self.scripts_lock.acquire()
        if script is not None:
            # Add the script and its location to the device's scripts list.
            self.scripts.append((script, location))
            # Notify any waiting worker threads that a new script is available.
            self.scripts_condition.notify_all()
            self.scripts_lock.release()
        else:
            # If script is None, signal that all scripts for this timestep have been assigned.
            self.scripts_assigned = True
            # Notify the main DeviceThread (and potentially workers) that script assignment is done.
            self.scripts_done_condition.notify_all()
            self.scripts_lock.release()

    def is_busy(self, location):
        """
        Checks if a specific location is currently marked as busy.

        This method is thread-safe using `location_busy_lock`.

        Args:
            location (int): The identifier of the location to check.

        Returns:
            bool: True if the location is busy, False otherwise.
        """
        self.location_busy_lock.acquire()
        ret = location in self.location_busy and self.location_busy[location]
        self.location_busy_lock.release()
        return ret

    def set_busy(self, location, value):
        """
        Sets the busy status of a specific location.

        This method is thread-safe using `location_busy_lock`.

        Args:
            location (int): The identifier of the location to update.
            value (bool): The new busy status (True for busy, False for free).
        """
        self.location_busy_lock.acquire()
        self.location_busy[location] = value
        self.location_busy_lock.release()

    def has_data(self, location):
        """
        Checks if the device has sensor data for a given location.

        This method is thread-safe using `data_lock`.

        Args:
            location (int): The identifier of the location to check.

        Returns:
            bool: True if data exists for the location, False otherwise.
        """
        self.data_lock.acquire()
        ret = location in self.sensor_data
        self.data_lock.release()
        return ret

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        This method is thread-safe using `data_lock`.

        Args:
            location (int): The identifier for the location for which to retrieve data.

        Returns:
            Any: The sensor data if the location exists, otherwise None.
        """
        self.data_lock.acquire()
        ret = self.sensor_data[location] if location in self.sensor_data else None
        self.data_lock.release()
        return ret

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        This method is thread-safe using `data_lock`.

        Args:
            location (int): The identifier for the location where the data should be set.
            data (Any): The new data to set for the location.
        """
        self.data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.data_lock.release()

    def shutdown(self):
        """
        Shuts down the device by joining its associated main thread and all worker threads.
        This ensures all background processes complete gracefully.
        """
        self.thread.join()
        for thread in self.worker_threads:
            thread.join()

class ScriptWorker(Thread):
    """
    A worker thread responsible for fetching and executing scripts for a Device.

    Each `ScriptWorker` continuously attempts to retrieve a script from its
    associated `Device`, manages location-specific access using conditions,
    collects data from neighbors, executes the script, and updates sensor data.
    """
    def __init__(self, device, index):
        """
        Initializes a new ScriptWorker instance.

        Args:
            device (Device): The Device object that this worker will serve.
            index (int): A unique index for this worker thread within the device.
        """
        Thread.__init__(self, name="Worker thread %d for device %d" % (index, device.device_id))
        self.device = device
        # Reference to the device's scripts_lock for accessing shared script state.
        self.lock = device.scripts_lock
        # Reference to the condition variable for signaling script completion.
        self.done_condition = device.scripts_done_condition
        # Reference to the condition variable for waiting on new scripts.
        self.condition = device.scripts_condition

    def run(self):
        """
        The main execution loop for the ScriptWorker.

        It continuously acquires the `scripts_lock`, waits for scripts to be
        enabled or available, fetches a script, releases the lock to execute it,
        then re-acquires the lock to update completion status and notify.
        This loop terminates when the device's `thread_running` flag is false.
        """
        # Acquire the scripts_lock to manage access to shared script queues and conditions.
        self.lock.acquire()

        while self.device.thread_running:
            script = None

            # Block Logic: Wait for scripts to be enabled or available.
            # If scripts are not enabled or all available scripts have been started,
            # the worker waits on the `scripts_condition`.
            if not self.device.scripts_enabled or \
                   self.device.scripts_started_idx >= len(self.device.scripts):
                self.condition.wait() # Releases lock and waits. Re-acquires on notification.
                continue # Re-evaluate conditions after waking up.

            # Get the next script to process.
            script = self.device.scripts[self.device.scripts_started_idx]
            # Increment the index of started scripts.
            self.device.scripts_started_idx = self.device.scripts_started_idx + 1
            # Notify other waiting workers that a script has been taken.
            self.condition.notify_all()

            # Release the lock before executing the script to avoid holding it during long computations.
            self.lock.release()
            # Execute the script.
            self.run_script(script[0], script[1])
            # Re-acquire the lock to update shared state after script execution.
            self.lock.acquire()

            # Increment the count of completed scripts.
            self.device.scripts_done_idx = self.device.scripts_done_idx + 1
            # Notify `scripts_done_condition` (typically the DeviceThread) that a script has finished.
            self.done_condition.notify_all()

        # Release the lock before exiting the thread.
        self.lock.release()


    def run_script(self, script, location):
        """
        Executes a single assigned script for a specific location.

        This method coordinates data access using location-specific conditions
        to ensure mutual exclusion and proper sequencing.

        Args:
            script (object): The script object to execute.
            location (int): The identifier of the location for which the script is run.
        """
        # Acquire the condition lock for this specific location.
        # This lock ensures that only one worker can process data for this location at a time.
        self.device.location_conditions[location].acquire()

        # If there are no neighbors, there's no data to collect, so release and return.
        if self.device.neighbours is None:
            self.device.location_conditions[location].release()
            return

        script_devices = [] # List of devices that have data for this location.
        # Collect devices that have data for the current location.
        for device in self.device.neighbours:
            if device.has_data(location):
                script_devices.append(device)
        if self.device.has_data(location):
            script_devices.append(self.device)

        # If no devices have data for this location, release the lock and return.
        if len(script_devices) == 0:
            self.device.location_conditions[location].release()
            return

        # Block Logic: Wait until all involved locations are not busy.
        # Invariant: Loop continues as long as any relevant device reports the location as busy.
        while True:
            free = True
            for device in script_devices:
                if device.is_busy(location):
                    free = False
                    break
            if free:
                break
            # If not all locations are free, wait on the condition variable.
            self.device.location_conditions[location].wait()

        # Collect data from all relevant devices and mark their locations as busy.
        script_data = []
        for device in script_devices:
            device.set_busy(location, True)
            script_data.append(device.get_data(location))
        # Notify any waiting threads that busy status might have changed.
        self.device.location_conditions[location].notify_all()

        # Release the location condition lock to allow script execution outside critical section.
        self.device.location_conditions[location].release()
        # Execute the script with the collected data.
        result = script.run(script_data)
        # Re-acquire the location condition lock for updating data.
        self.device.location_conditions[location].acquire()

        # Update data in all relevant devices and mark their locations as free.
        for device in script_devices:
            device.set_data(location, result)
            device.set_busy(location, False)
        # Notify any waiting threads that data has been updated and busy status changed.
        self.device.location_conditions[location].notify_all()

        # Release the location condition lock.
        self.device.location_conditions[location].release()


class DeviceThread(Thread):
    """
    Manages the overall timestep progression and coordinates script processing
    for a Device within the simulation.

    This thread ensures that all devices synchronize at each timestep,
    manages the lifecycle of script execution by enabling and disabling
    worker threads, and handles the collection of neighbor information.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread, managing timesteps.

        It continuously performs the following steps:
        1. Waits until the device is ready to start (initial setup complete).
        2. Waits at the shared `timestep_barrier` to synchronize with other devices.
        3. Retrieves updated neighbor information from the supervisor.
        4. If no neighbors are returned (e.g., simulation end), the loop breaks.
        5. Acquires `scripts_lock`, resets script progress counters, enables script
           processing for worker threads, and notifies them.
        6. Releases `scripts_lock`.
        7. Re-acquires `scripts_lock` and waits until all assigned scripts for
           the current timestep are completed by worker threads.
        8. Disables script processing and releases `scripts_lock`.
        9. Upon loop termination, it acquires `scripts_lock`, sets `thread_running`
           to False, and notifies all worker threads to shut down.
        """
        # Wait until the device's initial setup is complete and it's ready to start.
        self.device.ready_to_start.wait()

        while True:
            # Wait at the shared barrier to synchronize with all other devices for the new timestep.
            self.device.timestep_barrier.wait()

            # Retrieve updated neighbor information from the supervisor for the current round.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            # If supervisor returns None, it signals the simulation to terminate.
            if self.device.neighbours is None:
                break

            # Acquire scripts_lock to manage shared script state.
            self.device.scripts_lock.acquire()
            # Reset script progress counters for the new timestep.
            self.device.scripts_started_idx = 0
            self.device.scripts_done_idx = 0
            # Enable worker threads to start processing scripts.
            self.device.scripts_enabled = True
            # Notify all waiting worker threads that new scripts are available and enabled.
            self.device.scripts_condition.notify_all()
            self.device.scripts_lock.release() # Release lock after updating shared state.

            # Acquire scripts_lock to wait for all scripts to complete.
            self.device.scripts_lock.acquire()
            # Block Logic: Wait until all scripts assigned for this timestep are processed.
            # Invariant: Loop continues as long as not all scripts are assigned
            # (which means scripts are still being added) or not all started scripts are done.
            while not self.device.scripts_assigned or \
                  self.device.scripts_done_idx < len(self.device.scripts):
                self.device.scripts_done_condition.wait() # Releases lock and waits.
            # Disable script processing for worker threads after all are done.
            self.device.scripts_enabled = False
            self.device.scripts_lock.release() # Release lock after updating shared state.

        # Upon breaking from the main loop (simulation termination):
        self.device.scripts_lock.acquire() # Acquire lock to safely update shutdown state.
        self.device.thread_running = False # Signal worker threads to terminate.
        self.device.scripts_condition.notify_all() # Notify all waiting workers to check thread_running.
        self.device.scripts_lock.release() # Release lock.
