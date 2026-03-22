


from threading import Event, Thread, Lock, RLock
from Queue import Queue


class Device(object):
    """
    Represents a single computational device in a simulated distributed system.
    Each device has a unique ID, manages its own sensor data with read-write locks,
    interacts with a central supervisor, and processes scripts using a queue-based
    multi-threading model.
    """
    
    no_cores = 8 # Class-level constant: Maximum number of worker threads per device.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device object.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing sensor readings or local data,
                                 keyed by location.
            supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        # Dictionary to store RLock (reentrant locks) for each sensor data location.
        # This allows multiple reads but exclusive writes to a specific location.
        self.sensor_data_locks = {}
        self.supervisor = supervisor
        # List of other devices in the system, to facilitate inter-device communication.
        self.devices_other = []

        self.scripts = [] # List to store (script, location) tuples assigned to this device.
        # Queue to hold scripts to be processed by DeviceThreads.
        self.script_queue = Queue()
        # Lock to protect access to the `scripts` list and `script_queue`.
        self.scripts_lock = Lock()
        # Virtual socket (Queue) for inter-device synchronization signals.
        self.virt_socket = Queue()

        # Lock for synchronizing the start of a timepoint's processing.
        self.start_lock = Lock()
        # Flag indicating if the device is at the start of a new timepoint's processing.
        self.start_is_at = True
        # Event to signal when all processing for a timepoint is complete.
        self.end_event = Event()

        self.neighbours = [] # List of neighboring devices.
        self.counter = 1     # Counter used in synchronization for managing timepoint completion.

        self.threads = []    # List of worker DeviceThread instances.
        # Block Logic: Initializes a fixed number of DeviceThread instances.
        for _ in range(Device.no_cores):
            self.threads.append(DeviceThread(self))
        self.active_threads = 1 # Number of currently active worker threads.
        self.threads[0].start() # Starts the first worker thread immediately.

    def __start_thread(self):
        """
        Starts an additional worker thread if the number of active threads
        is less than `no_cores` and the number of pending scripts exceeds
        the current active threads.
        """
        # Conditional Logic: Prevents starting more threads than the maximum allowed.
        if self.active_threads >= Device.no_cores:
            return

        no_thr = len(self.scripts) # Number of assigned scripts.
        # Conditional Logic: If there are more scripts than active threads, start a new thread.
        if no_thr > self.active_threads:
            self.threads[self.active_threads].start()
            self.active_threads += 1

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the device's list of other devices and initializes RLocks
        for each sensor data location.

        Args:
            devices (List[Device]): A list of all Device instances in the system.
        """
        # Sets the list of other devices (excluding self).
        self.devices_other = devices
        self.devices_other.remove(self)
        # Block Logic: Initializes an RLock for each sensor data location.
        for loc in self.sensor_data:
            self.sensor_data_locks[loc] = RLock()

    def sync_send(self):
        """
        Sends a synchronization signal through the virtual socket.
        """
        self.virt_socket.put(None)

    def sync_devices(self):
        """
        Synchronizes with all other devices by sending and receiving signals
        through virtual sockets. This acts as a barrier for global synchronization.
        """
        # Block Logic: Sends a signal to all other devices.
        for dev in self.devices_other:
            dev.sync_send()
        # Block Logic: Waits to receive a signal from all other devices.
        for _ in self.devices_other:
            _ = self.virt_socket.get()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a specific location.
        If script is None, it signals the worker threads that script assignment
        for the current timepoint is complete.

        Args:
            script (Script or None): The script object to execute, or None for signaling.
            location (int): The logical location associated with the script's execution.
        """
        # Conditional Logic: If a script object is provided, add it to the script queue and list.
        if script is not None:
            with self.scripts_lock: # Protects access to scripts list and queue.
                self.script_queue.put((script, location))
                self.scripts.append((script, location))
                self.__start_thread() # Potentially starts a new worker thread.
        else:
            # If script is None, put `None` into the queue for each active thread to signal completion.
            with self.scripts_lock:
                for _ in range(self.active_threads):
                    self.script_queue.put(None)

    def get_data(self, location):
        """
        Retrieves sensor data for a given location. Access is protected by an RLock.

        Args:
            location (int): The key for the sensor data.

        Returns:
            Any: The data associated with the location, or None if not found.
        """
        if location in self.sensor_data:
            with self.sensor_data_locks[location]: # Acquire RLock for this location.
                return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location. Access is protected by an RLock.

        Args:
            location (int): The key for the sensor data.
            data (Any): The new data to set for the location.
        """
        if location in self.sensor_data:
            with self.sensor_data_locks[location]: # Acquire RLock for this location.
                self.sensor_data[location] = data

    def get_data_lock(self, location):
        """
        Acquires the RLock for a given location and then retrieves the sensor data.
        The lock must be released separately using `set_data_unlock`.

        Args:
            location (int): The key for the sensor data.

        Returns:
            Any: The data associated with the location, or None if not found.
        """
        if location in self.sensor_data:
            self.sensor_data_locks[location].acquire() # Acquire RLock.
            return self.sensor_data[location]
        return None

    def set_data_unlock(self, location, data):
        """
        Sets or updates sensor data for a given location and then releases its RLock.
        This is used in conjunction with `get_data_lock`.

        Args:
            location (int): The key for the sensor data.
            data (Any): The new data to set for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.sensor_data_locks[location].release() # Release RLock.

    def timepoint_init(self):
        """
        Initializes the device for a new timepoint. This includes repopulating
        the script queue and updating the list of neighbors from the supervisor.
        """
        with self.scripts_lock: # Protects access to scripts list and queue.
            # Block Logic: Repopulates the script queue with all assigned scripts for the next timepoint.
            for script in self.scripts:
                self.script_queue.put(script)
        # Fetches the list of neighboring devices from the supervisor.
        self.neighbours = self.supervisor.get_neighbours()
        # Conditional Logic: If neighbors exist, convert to list and sort by device_id.
        if self.neighbours is not None:
            self.neighbours = list(self.neighbours)
            self.neighbours.append(self) # Include self in the list of neighbors for data collection.
            
            self.neighbours.sort(key=lambda x: x.device_id) # Sorts neighbors by device_id.

    def shutdown(self):
        """
        Shuts down the device by joining its operational threads, ensuring all
        tasks are completed before the program exits.
        """
        # Block Logic: Joins all worker threads to ensure their termination.
        for thr in self.threads:
            if thr.isAlive(): # Only join if the thread is still running.
                thr.join()


class DeviceThread(Thread):
    """
    The main operational thread for a Device. It continuously processes scripts
    from the device's script queue, interacts with neighboring devices,
    and participates in timepoint-based synchronization.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The Device instance that this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.
        It manages the lifecycle of script processing within a timepoint,
        including initialization, script execution (gathering data with locks,
        running scripts, updating data with locks), and inter-device synchronization.
        """
        while True:
            # Block Logic: Synchronizes the start of timepoint processing among threads.
            with self.device.start_lock:
                # Conditional Logic: If this thread is the first to arrive at `start_lock`,
                # it initializes the timepoint.
                if self.device.start_is_at:
                    self.device.start_is_at = False # Mark that timepoint initialization has started.
                    self.device.timepoint_init()    # Initialize scripts queue and neighbors.
                    self.device.end_event.clear()   # Clear the end event for the new timepoint.

            # Fetches the current list of neighbors.
            neighbours = self.device.neighbours
            # Conditional Logic: If no neighbors are returned (e.g., system shutdown), exit loop.
            if neighbours is None:
                break

            # Block Logic: Processes scripts from the script queue until a `None` sentinel is received.
            while True:
                # Gets a script-location pair from the queue. This call blocks until an item is available.
                pair = self.device.script_queue.get()
                # Conditional Logic: If `None` is received, it signifies no more scripts for this timepoint.
                if pair is None:
                    self.device.script_queue.task_done() # Mark task as done for the queue.
                    break
                script = pair[0]   # The script to execute.
                location = pair[1] # The location associated with the script.

                script_data = [] # Data collected for the script.
                
                # Block Logic: Collects data from the device and its neighbors for the specified location.
                # It acquires a lock for each device's data before reading.
                for device in neighbours:
                    data = device.get_data_lock(location) # Acquire lock and get data.
                    if data is not None:
                        script_data.append(data)

                # Conditional Logic: If there is any data, execute the script.
                if script_data != []:
                    # Executes the script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Updates the data on the device and its neighbors with the script's result.
                    # It releases the lock for each device's data after writing.
                    for device in neighbours:
                        device.set_data_unlock(location, result)

                self.device.script_queue.task_done() # Mark task as done for the queue.

            # Block Logic: Waits until all tasks in the script queue are marked as done.
            self.device.script_queue.join()

            # Block Logic: Synchronizes the end of timepoint processing among threads.
            with self.device.start_lock:
                # Conditional Logic: If this thread was the one that initialized the timepoint,
                # it will signal completion after all other threads have also completed.
                if not self.device.start_is_at:
                    self.device.start_is_at = True      # Reset flag for next timepoint.
                    self.device.sync_devices()          # Synchronize with other devices.
                    
                    self.device.counter = self.device.active_threads - 1 # Initialize counter.
                else:
                    self.device.counter = self.device.counter - 1 # Decrement counter.
                # Conditional Logic: If this is the last thread to complete its timepoint processing.
                if self.device.counter == 0:
                    self.device.end_event.set() # Signal that the timepoint is fully done.
            self.device.end_event.wait() # Wait until the timepoint is fully done by all threads.

