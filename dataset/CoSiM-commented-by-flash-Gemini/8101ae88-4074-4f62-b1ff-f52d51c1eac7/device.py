


from threading import Event, Thread, Lock, Condition

class Device(object):
    """
    Represents a single computational device in a simulated distributed system.
    Each device has an ID, sensor data, and communicates with a supervisor.
    It can receive and execute scripts, leveraging a pool of `MyThread` workers
    managed by its `DeviceThread`. Synchronization across devices is handled
    by a shared barrier and per-location locks.
    """

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
        self.supervisor = supervisor
        # Event to signal when a script has been assigned to this device.
        self.script_received = Event()
        self.scripts = [] # List to store (script, location) tuples assigned to this device.
        # Event to signal when all scripts for the current timepoint are done.
        self.timepoint_done = Event()

        # The main thread for this device, which handles its operational lifecycle.
        self.thread = DeviceThread(self)
        self.thread.start()
        # List to keep track of all unique locations across devices.
        self.locations = []
        # Lock for synchronizing access to shared data (e.g., in `get_data_lock`, `set_data_unlock`).
        self.sync_data_lock = Lock()
        # Dictionary of locks, where each lock protects data at a specific location.
        # This will be shared across all devices for per-location data access control.
        self.sync_location_lock = {}
        # Number of CPU cores or worker threads to utilize per device for script execution.
        self.cores = 8
        # Barrier for synchronization with other devices. Initialized to None,
        # will be properly configured by setup_devices.
        self.barrier = None

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the shared barrier and per-location locks for all devices.
        Only device with ID 0 initializes these shared resources.

        Args:
            devices (List[Device]): A list of all Device instances in the system.
        """
        # Block Logic: Only the device with ID 0 initializes the shared resources (locks, barrier).
        if self.device_id == 0:
            # Determine the total number of unique locations across all devices.
            locations_number = self.get_locations_number(devices)
            # Block Logic: Initializes a Lock for each unique location.
            for location in range(locations_number):
                self.sync_location_lock[location] = Lock()
            # Initializes a ReusableBarrierCond with the total number of devices.
            self.barrier = ReusableBarrierCond(len(devices))
            # Block Logic: Distributes the initialized shared resources to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.sync_location_lock = self.sync_location_lock

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a specific location.
        If script is None, it signals that the timepoint's script assignments are done.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The logical location (index) associated with the script's execution.
        """
        # Conditional Logic: If a script object is provided, add it to the device's script list
        # and signal that scripts have been received.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # If script is None, it means no more scripts for this timepoint,
            # so signal completion for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The key (index) for the sensor data.

        Returns:
            Any: The data associated with the location, or None if not found.
        """
        # Conditional Logic: Checks if the location exists in sensor_data before accessing.
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (int): The key (index) for the sensor data.
            data (Any): The new data to set for the location.
        """
        # Conditional Logic: Updates data only if the location already exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its operational thread, ensuring all
        tasks are completed before the program exits.
        """
        self.thread.join()

    def get_locations_number(self, devices):
        """
        Helper method to determine the total number of unique locations across all devices.

        Args:
            devices (List[Device]): A list of all Device instances in the system.

        Returns:
            int: The total count of unique locations.
        """
        # Block Logic: Iterates through all devices and their sensor data to collect
        # all unique location keys.
        for device in devices:
            for location in device.sensor_data:
                if location not in self.locations:
                    self.locations.append(location)
        return len(self.locations)



class DeviceThread(Thread):
    """
    The main operational thread for a Device. It drives the device's behavior,
    synchronizing with other devices via a barrier and distributing assigned scripts
    to a pool of `MyThread` workers for parallel execution.
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
        It repeatedly fetches neighbors, waits for scripts to be assigned,
        creates and manages a pool of `MyThread`s to execute scripts in parallel,
        and then synchronizes with other devices using a barrier.
        """
        while True:
            # Fetches the current list of neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (e.g., system shutdown), exit loop.
            if neighbours is None:
                break
            
            # Block Logic: Waits for the supervisor to signal that the current timepoint
            # processing is done (i.e., all scripts for this timepoint have been assigned).
            self.device.timepoint_done.wait()
            
            my_threads = [] # List to hold MyThread instances.
            num_threads = self.device.cores # Number of worker threads to use.
            index = 0 # Used for round-robin assignment of scripts to threads.
            # Conditional Logic: If `my_threads` is empty, initialize the worker thread pool.
            if len(my_threads) == 0:
                # Block Logic: Creates `num_threads` instances of MyThread workers.
                for i in range(num_threads):
                    thread = MyThread(self)
                    my_threads.append(thread)
            
            # Block Logic: Assigns scripts to the worker threads in a round-robin fashion.
            for (script, location) in self.device.scripts:
                my_threads[index % num_threads].assign_script(script, location)
                index = index + 1
            
            # Block Logic: Starts all worker threads and then waits for their completion.
            for i in range(num_threads):
                my_threads[i].set_neighbours(neighbours) # Sets neighbors for each thread.
                my_threads[i].start()
            for i in range(num_threads):
                my_threads[i].join() # Waits for each worker thread to finish.
            
            self.device.timepoint_done.clear() # Clear the event for the next timepoint.
            
            # Block Logic: Waits at the shared barrier for all DeviceThreads to reach this point.
            self.device.barrier.wait()

class MyThread(Thread):
    """
    A worker thread responsible for executing a subset of scripts assigned to its parent device.
    It gathers data from the parent device and its neighbors for specific locations,
    runs the assigned scripts, and then updates the data on the involved devices,
    ensuring thread-safe access to data locations.
    """
    
    def __init__(self, parent_device_thread):
        """
        Initializes a MyThread.

        Args:
            parent_device_thread (DeviceThread): A reference to the parent DeviceThread.
        """
        Thread.__init__(self)
        self.parent = parent_device_thread
        self.scripts = [] # List to store (script, location) tuples assigned to this thread.
        self.neighbours = [] # List of neighboring devices.

    def set_neighbours(self, neighbours):
        """
        Sets the list of neighboring devices for this thread to use during script execution.

        Args:
            neighbours (List[Device]): A list of neighboring Device instances.
        """
        self.neighbours = neighbours

    def assign_script(self, script, location):
        """
        Assigns a single script and its location to this worker thread.

        Args:
            script (Script): The script object to execute.
            location (int): The logical location associated with the script's execution.
        """
        self.scripts.append((script, location))

    def run(self):
        """
        The main execution method for MyThread.
        It iterates through its assigned scripts, acquires location-specific locks,
        gathers data from devices, runs the script, updates data, and releases locks.
        """
        # Block Logic: Iterates through each script assigned to this worker thread.
        for (script, location) in self.scripts:
            # Block Logic: Acquire the location-specific lock to protect data access for this location.
            self.parent.device.sync_location_lock[location].acquire()
            script_data = [] # Data collected for the script.
            
            # Block Logic: Collects data from neighboring devices for the specified location.
            for device in self.neighbours:
                # Acquire the device's general data lock before getting data.
                device.sync_data_lock.acquire()
                data = device.get_data(location)
                device.sync_data_lock.release() # Release the device's general data lock.
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Collects data from the current device for the specified location.
            self.parent.device.sync_data_lock.acquire()
            data = self.parent.device.get_data(location)
            self.parent.device.sync_data_lock.release()
            if data is not None:
                script_data.append(data)

            # Conditional Logic: If there is any data, execute the script.
            if script_data != []:
                # Executes the script with the collected data.
                result = script.run(script_data)
                
                # Block Logic: Updates the data on neighboring devices with the script's result.
                for device in self.neighbours:
                    device.sync_data_lock.acquire() # Acquire the device's general data lock.
                    device.set_data(location, result)
                    device.sync_data_lock.release() # Release the device's general data lock.

                # Block Logic: Updates the data on the current device with the script's result.
                self.parent.device.sync_data_lock.acquire()
                self.parent.device.set_data(location, result)
                self.parent.device.sync_data_lock.release()
            
            # Release the location-specific lock.
            self.parent.device.sync_location_lock[location].release()


class ReusableBarrierCond(object):
    """
    Implements a reusable barrier synchronization mechanism using a Condition object.
    This barrier allows a fixed number of threads (`num_threads`) to wait for each other
    at a synchronization point, and then allows them to proceed. It is "reusable"
    because it can be used multiple times without reinitialization.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrierCond.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any of them can proceed.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Counter for threads at the barrier.
        self.cond = Condition() # The condition variable for synchronization.

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have also called `wait()`. Once all threads have arrived, they are all released.
        """
        self.cond.acquire() # Acquire the condition's lock.
        self.count_threads -= 1
        # Conditional Logic: If this is the last thread to arrive at the barrier.
        if self.count_threads == 0:
            self.cond.notify_all() # Release all waiting threads.
            self.count_threads = self.num_threads # Reset the counter for next use.
        else:
            self.cond.wait() # Wait until notified by the last thread.
        self.cond.release() # Release the condition's lock.
