


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    Implements a reusable barrier synchronization mechanism using semaphores.
    This barrier allows a fixed number of threads (`num_threads`) to wait for each other
    at a synchronization point, and then allows them to proceed. It is "reusable"
    because it can be used multiple times without reinitialization.
    It uses a two-phase approach to prevent threads from "slipping" through the barrier
    if they arrive too early for the next cycle.
    The counters `count_threads1` and `count_threads2` are implemented as single-element lists
    to allow modification within nested scopes (e.g., `with self.count_lock:`).
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrierSem.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any of them can proceed.
        """
        self.num_threads = num_threads
        # Counter for threads in phase 1, wrapped in a list for mutable reference.
        self.count_threads1 = [self.num_threads]
        # Counter for threads in phase 2, wrapped in a list for mutable reference.
        self.count_threads2 = [self.num_threads]
        # Lock to protect access to the thread counters.
        self.count_lock = Lock()
        # Semaphore for threads waiting in phase 1. Initialized to 0, so all threads
        # will block until released.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for threads waiting in phase 2. Initialized to 0.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have also called `wait()`. Once all threads have arrived, they are all released.
        This method executes both phase1 and phase2 of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the two-phase barrier synchronization.

        Args:
            count_threads (List[int]): A list containing the counter for the current phase.
                                       It's a list so its value can be modified by reference.
            threads_sem (Semaphore): The semaphore associated with this phase, used to
                                     block and release threads.
        """
        with self.count_lock: # Protect access to the counter.
            count_threads[0] -= 1
            # Conditional Logic: If this is the last thread to arrive at the current phase.
            if count_threads[0] == 0:
                # Release all `num_threads` from the semaphore.
                i = 0
                while i < self.num_threads:
                    threads_sem.release()
                    i = i + 1
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire() # Block until released by the last thread in this phase.


class Device(object):
    """
    Represents a single computational device in a simulated distributed system.
    Each device has an ID, sensor data, and communicates with a supervisor.
    It can receive and execute scripts, synchronizing with other devices using two barriers
    and managing data access with global and per-location locks.
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
        self.scripts = [] # List to store (script, location) tuples assigned to this device.
        # Event to signal when the device has completed processing for a timepoint.
        self.timepoint_done = Event()
        # Barrier for synchronization with other devices (first phase).
        self.barrier = None
        # Second barrier for synchronization with other devices (second phase).
        self.barrier2 = None
        self.max_threads = 8 # Maximum number of dynamically spawned threads for script execution.
        # The main thread for this device, which handles its operational lifecycle.
        self.thread = DeviceThread(self)
        # Lock to protect shared barrier initialization among devices.
        self.barrier_lock = Lock()
        # Dictionary to store per-location locks. Initialized for existing sensor_data keys.
        self.location_locks = {}
        # Block Logic: Initializes `location_locks` with `None` for existing sensor data locations.
        for loc in sensor_data.keys():
            self.location_locks[loc] = None
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the shared barriers and per-location locks for all devices.
        Only the device with ID 0 initializes these shared resources.

        Args:
            devices (List[Device]): A list of all Device instances in the system.
        """
        index = 0
        
        # Block Logic: Ensures all devices share the same barrier_lock.
        self.barrier_lock = devices[index].barrier_lock

        # Block Logic: Acquire the barrier_lock to ensure atomic initialization of shared resources.
        self.barrier_lock.acquire()
        
        # Lists to check if barriers have already been initialized by another device.
        barrier_list = [d.barrier for d in devices if d.barrier is not None]
        barrier_list2 = [d.barrier2 for d in devices if d.barrier2 is not None]
        # Collects all location_locks dictionaries from other devices.
        loc_list = [device.location_locks for device in devices]

        
        # Block Logic: Populates `self.location_locks` with unique locks found across all devices.
        for loc in loc_list:
            for val in loc.keys():
                # Conditional Logic: If a lock for a location is not yet set, use the one from `loc`.
                if val not in self.location_locks.keys():
                    self.location_locks[val] = loc[val]
                elif loc[val] is not None and self.location_locks[val] is None:
                    self.location_locks[val] = loc[val]

        
        keys = self.location_locks.keys()
        rest = [index for index in keys if self.location_locks[index] is None]
        for index in rest:
            self.location_locks[index] = Lock()

        
        index = 0
        if len(barrier_list) == 0: # If no barriers have been initialized yet.
            self.barrier = ReusableBarrierSem(len(devices))
            self.barrier2 = ReusableBarrierSem(len(devices))
        else: # Otherwise, use the existing barriers.
            self.barrier = barrier_list[index]
            self.barrier2 = barrier_list2[index]

        self.barrier_lock.release() # Release the barrier_lock.

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a specific location.
        If script is None, it signals that the timepoint's script assignments are done.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The logical location (index) associated with the script's execution.
        """
        # Conditional Logic: If a script object is provided, add it to the device's script list.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If script is None, it means no more scripts for this timepoint,
            # so signal completion for the script assignment phase and the timepoint.
            self.barrier.wait() # All devices signal script assignment completion.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The key (index) for the sensor data.

        Returns:
            Any: The data associated with the location, or None if not found.
        """
        result = None
        if location in self.sensor_data:
            result = self.sensor_data[location]
        return result

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (int): The key (index) for the sensor data.
            data (Any): The new data to set for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its operational thread, ensuring all
        tasks are completed before the program exits.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    The main operational thread for a Device. It drives the device's behavior,
    synchronizing with other devices via two barriers and executing assigned scripts
    by dynamically spawning child `Thread`s (`DeviceSubThread`).
    """
    

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The Device instance that this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.num_threads = 1 # Counter for currently active child threads.

    def run_scripts(self, script, location, neighbours):
        """
        Executes a single script, gathering data from the device and its neighbors,
        and then updates the data on the involved devices. Access to specific
        locations is protected by `location_locks`.

        Args:
            script (Script): The script object to be executed.
            location (int): The logical location (index) associated with this script.
            neighbours (List[Device]): A list of neighboring Device instances.
        """
        script_data = []

        # Block Logic: Acquire the location-specific lock to protect data access for this location.
        self.device.location_locks[location].acquire()

        # Block Logic: Collects data from neighboring devices for the specified location.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collects data from the current device for the specified location.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        # Conditional Logic: If there is any data, execute the script.
        if script_data != []:
            result = script.run(script_data) # Execute the script.
            
            # Block Logic: Updates the data on neighboring devices with the script's result.
            for device in neighbours:
                device.set_data(location, result)
            
            # Block Logic: Updates the data on the current device with the script's result.
            self.device.set_data(location, result)

        # Release the location-specific lock.
        self.device.location_locks[location].release()

    def run(self):
        """
        The main execution loop for the DeviceThread.
        It repeatedly fetches neighbors, synchronizes with other devices using barriers,
        dynamically spawns `DeviceSubThread`s to execute scripts in parallel (up to `max_threads`),
        and waits for their completion.
        """
        while True:
            # Fetches the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (e.g., system shutdown),
            # synchronize one last time and exit the loop.
            if neighbours is None:
                self.device.barrier.wait()
                break
            # Block Logic: First synchronization point, ensuring all devices are ready for the timepoint.
            self.device.barrier.wait()
            
            child_threads = [] # List to hold dynamically spawned child threads.
            
            # Block Logic: Iterates through assigned scripts and either executes them directly
            # or spawns new threads if the thread limit (`max_threads`) has not been reached.
            for (script, location) in self.device.scripts:
                # Conditional Logic: If current active threads are below max, spawn a new thread.
                if self.num_threads < self.device.max_threads:
                    self.num_threads = self.num_threads + 1
                    arguments = (script, location, neighbours)
                    child = Thread(target=self.run_scripts, args=arguments)
                    child_threads.append(child)
                    child.start()
                else: # Otherwise, execute the script directly in the current thread.
                    self.run_scripts(script, location, neighbours)
            # Block Logic: Waits for all dynamically spawned child threads to complete.
            for child in child_threads:
                
                child.join()
                self.num_threads = self.num_threads - 1 # Decrement count of active child threads.
            
            # Block Logic: Second synchronization point, ensuring all devices have finished script execution.
            self.device.barrier2.wait()
            # Waits for the timepoint to be explicitly marked as done.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Clear the event for the next timepoint.


class DeviceSubThread(Thread):
    """
    A worker thread dynamically spawned by `DeviceThread` to execute a single script.
    It gathers data from the parent device and its neighbors at a specific location,
    runs the assigned script, and then updates the data on the involved devices,
    ensuring thread-safe access to data locations using per-location locks.
    """
    
    def __init__(self, devicethread, neighbours, script, location):
        """
        Initializes a DeviceSubThread.

        Args:
            devicethread (DeviceThread): A reference to the parent DeviceThread.
            neighbours (List[Device]): A list of neighboring Device instances
                                       whose data might be relevant.
            script (Script): The script object to be executed.
            location (int): The logical location (index) associated with this script.
        """
        Thread.__init__(self, name="Device SubThread %d"
            % devicethread.device.device_id)
        self.neighbours = neighbours
        self.devicethread = devicethread
        self.script = script
        self.location = location

    def run(self):
        """
        The main execution method for DeviceSubThread.
        It acquires a lock for its designated location, gathers data from the device
        and its neighbors, executes the script, and then updates the data on
        the device and its neighbors. Finally, it releases the location lock.
        """
        # Block Logic: Acquire the location-specific lock to protect data access for this location.
        self.devicethread.device.locationlock[self.location].acquire()
        script_data = [] # Data collected for the script.
        
        # Block Logic: Collects data from neighboring devices for the specified location.
        for device in self.neighbours:
            # Data access is done directly via `device.get_data`, which does not
            # explicitly acquire a lock. This means thread safety relies on
            # `locationlock` for exclusive write access and `device.lock` for
            # read-write synchronization if multiple subthreads attempt to read/write
            # the same location on the same device concurrently.
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collects data from the current device for the specified location.
        data = self.devicethread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Conditional Logic: If there is any data, execute the script.
        if script_data != []:
            result = self.script.run(script_data) # Execute the script.
            
            # Block Logic: Updates the data on neighboring devices with the script's result.
            for device in self.neighbours:
                # The `device.lock` is acquired here to protect the `set_data` call
                # on the neighboring device.
                with device.lock:
                    device.set_data(self.location, result)
            
            # Block Logic: Updates the data on the current device with the script's result.
            with self.devicethread.device.lock:
                self.devicethread.device.set_data(self.location, result)
        
        # Release the location-specific lock.
        self.devicethread.device.locationlock[self.location].release()
