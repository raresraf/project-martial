


from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier(object):
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
        Initializes the ReusableBarrier.

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
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire() # Block until released by the last thread in this phase.


class ScriptThread(Thread):
    """
    A thread dedicated to executing a single script for a specific device.
    It collects relevant data from the device itself and its neighbors,
    runs the script, and then updates the data on the involved devices,
    ensuring thread-safe access to data locations using per-location locks.
    """
    
    def __init__(self, script, location, device, neighbours):
        """
        Initializes a ScriptThread.

        Args:
            script (Script): The script object to be executed.
            location (int): The logical location (index) associated with this script.
            device (Device): The Device instance that owns this script thread.
            neighbours (List[Device]): A list of neighboring Device instances
                                       whose data might be relevant.
        """
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        The main execution method for ScriptThread.
        It acquires a lock for its designated location, gathers data from the device
        and its neighbors, executes the script, and then updates the data on
        the device and its neighbors. Finally, it releases the location lock.
        """
        # Block Logic: Acquire the lock for this specific location to protect data access.
        self.device.hash_locatie[self.location].acquire()

        script_data = [] # Data collected for the script.
        # Block Logic: Collects data from neighboring devices for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        # Block Logic: Collects data from the current device for the specified location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Conditional Logic: If there is any data to process, run the script.
        if script_data != []:
            result = self.script.run(script_data) # Execute the script.

            # Block Logic: Acquire the general device lock before updating data across devices.
            # This general lock might be redundant if per-location locks are sufficient,
            # but it ensures atomicity of update operations if multiple locations are involved.
            self.device.lock.acquire() 
            # Block Logic: Updates the data on neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.location, result)
            # Block Logic: Updates the data on the current device with the script's result.
            self.device.set_data(self.location, result)
            self.device.lock.release() # Release the general device lock.

        self.device.hash_locatie[self.location].release() # Release the location-specific lock.

class Device(object):
    """
    Represents a single computational device in a simulated distributed system.
    Each device has an ID, sensor data, and communicates with a supervisor.
    It can receive and execute scripts, synchronizing with other devices using a barrier
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
        # Event to signal when a script has been assigned to this device.
        self.script_received = Event()
        self.scripts = [] # List to store (script, location) tuples assigned to this device.
        # Event to signal when the device has completed processing for a timepoint.
        self.timepoint_done = Event()
        # The main thread for this device, which handles its operational lifecycle.
        self.thread = DeviceThread(self)
        # Barrier for synchronization with other devices. Initialized to None,
        # will be properly configured by setup_devices.
        self.barrier = None
        # General lock for the device, will be shared across devices.
        self.lock = None
        # Dictionary of locks, where each lock protects data at a specific location (hash_locatie in code).
        self.hash_locatie = None
        self.thread.start() # Start the device's operational thread.

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the shared barrier, global lock, and per-location locks for all devices.
        These shared resources are initialized by the first device (device_id == devices[0].device_id)
        and then distributed to all other devices.

        Args:
            devices (List[Device]): A list of all Device instances in the system.
        """
        # Block Logic: Only the first device in the list initializes the shared resources.
        if self.device_id == devices[0].device_id:
            barrier = ReusableBarrier(len(devices)) # Initialize shared barrier.
            my_lock = Lock() # Initialize shared general lock.
            hash_locatie = {} # Initialize dictionary for per-location locks.
            # Block Logic: Create a Lock for each possible location (up to 100).
            for i in range(101):
                hash_locatie[i] = Lock()
            self.barrier = barrier
            self.lock = my_lock
            self.hash_locatie = hash_locatie
            # Block Logic: Distribute the initialized shared resources to all devices.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
                if device.lock is None:
                    device.lock = my_lock
                if device.hash_locatie is None:
                    device.hash_locatie = hash_locatie

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

class DeviceThread(Thread):
    """
    The main operational thread for a Device. It drives the device's behavior,
    synchronizing with other devices via a barrier, and executing assigned
    scripts by spawning `ScriptThread`s.
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
        processes scripts via `ScriptThread`s, and synchronizes using a barrier.
        """
        # Block Logic: The main operational loop for the device.
        while True:
            # Pre-condition: Device is ready for a new timepoint/iteration.
            # Post-condition: `neighbours` contains the list of current neighboring devices.
            neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (e.g., system shutdown), exit loop.
            if neighbours is None:
                break

            # Block Logic: Waits for the supervisor to signal that the current timepoint
            # processing is done (i.e., all scripts for this timepoint have been assigned).
            self.device.timepoint_done.wait()

            script_list = [] # List to hold ScriptThread instances.

            # Block Logic: Creates a ScriptThread for each assigned script.
            for (script, location) in self.device.scripts:
                script_list.append(ScriptThread(script,
                                                location,
                                                self.device,
                                                neighbours))

            # Block Logic: Starts all ScriptThreads in parallel.
            for thread in script_list:
                thread.start()

            # Block Logic: Waits for all ScriptThreads to complete their execution.
            for thread in script_list:
                thread.join()

            # Clears the timepoint_done event for the next cycle.
            self.device.timepoint_done.clear()
            # Block Logic: Waits at the shared barrier for all DeviceThreads to reach this point.
            self.device.barrier.wait()
