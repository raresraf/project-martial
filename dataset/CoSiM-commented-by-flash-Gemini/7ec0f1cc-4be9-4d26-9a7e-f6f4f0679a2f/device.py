


from threading import *


class Device(object):
    """
    Represents a single computational device in a simulated distributed system.
    Each device has an ID, sensor data, and communicates with a supervisor.
    It can receive and execute scripts, synchronizing with other devices using a barrier
    and managing data access with per-location locks.
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
        self.script_received = Event() # This event is cleared/set in DeviceThread and assign_script
        self.scripts = [] # List to store (script, location) tuples assigned to this device.
        # Event to signal when the device has completed processing for a timepoint.
        self.timepoint_done = Event() # This event is cleared/set in DeviceThread and assign_script
        # The main thread for this device, which handles its operational lifecycle.
        self.thread = DeviceThread(self)
        self.thread.start() # Start the device's operational thread.

        # Lock to protect concurrent access to the device's sensor_data dictionary.
        self.lock_data = Lock()
        # List of locks, where each lock protects data at a specific location.
        # Initialized to an empty list and populated by setup_devices.
        self.lock_location = []
        # Barrier for synchronization with other devices. Initialized to None,
        # will be properly configured by setup_devices.
        self.time_barrier = None

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the shared barrier and per-location locks for inter-device synchronization.
        Only device with ID 0 initializes these shared resources.

        Args:
            devices (List[Device]): A list of all Device instances in the system.
        """
        # Block Logic: Only the device with ID 0 initializes the shared barrier and locks.
        if self.device_id == 0:
            # Initializes a ReusableBarrierSem with the total number of devices.
            self.time_barrier = ReusableBarrierSem(len(devices)) 

            # All devices get a reference to this shared barrier.
            for device in devices:
                device.time_barrier = self.time_barrier

            loc_num = 0 # Tracks the maximum location index.
            # Block Logic: Determines the highest location index across all devices
            # to prepare for creating per-location locks.
            for device in devices:
                for location in device.sensor_data:
                    loc_num = max(loc_num, location) # Assuming location keys are integers.
            # Block Logic: Creates a Lock for each possible location index.
            for i in range(loc_num + 1):
                self.lock_location.append(Lock()) 

            # All devices get a reference to this shared list of location-specific locks.
            for device in devices:
                device.lock_location = self.lock_location 

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a specific location.
        If script is None, it signals that no more scripts will be assigned for the current
        timepoint, and the device should proceed.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The logical location (index) associated with the script's execution.
        """
        # Conditional Logic: If a script is provided, append it to the list and set script_received event.
        if script is not None:
            self.scripts.append((script, location))
            # Signals to the DeviceThread that new scripts are available.
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
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location. Access to `sensor_data` is
        protected by `lock_data` to ensure thread safety.

        Args:
            location (int): The key (index) for the sensor data.
            data (Any): The new data to set for the location.
        """
        with self.lock_data: # Acquire lock to protect access to `sensor_data`.
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
    The main operational thread for a Device. It continuously manages the device's lifecycle,
    including synchronizing with other devices, executing assigned scripts via SlaveThreads,
    and waiting for new instructions.
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
        It repeatedly fetches neighbors, waits for timepoint completion,
        processes assigned scripts by spawning `SlaveThread`s, joins them,
        and synchronizes with other devices using a shared barrier.
        """
        while True:
            slaves = [] # List to hold SlaveThread instances.
            
            # Pre-condition: Device is ready for a new timepoint/iteration.
            # Post-condition: `neighbours` contains the list of current neighboring devices.
            neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (e.g., system shutdown), exit loop.
            if neighbours is None:
                break

            # Block Logic: Waits for the supervisor to signal that the current timepoint
            # processing is done (i.e., all scripts for this timepoint have been assigned).
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Reset the event for the next timepoint.

            # Block Logic: For each assigned script, create and start a SlaveThread.
            for (script, location) in self.device.scripts:
                slave = SlaveThread(script, location, neighbours, self.device) 
                slaves.append(slave)
                slave.start()

            # Block Logic: Waits for all spawned SlaveThreads to complete their execution.
            for i in range(len(slaves)):
                slaves.pop().join()

            # Block Logic: Synchronizes with other DeviceThreads at a shared barrier.
            # This ensures all devices complete their script processing before proceeding.
            self.device.time_barrier.wait() 

class SlaveThread(Thread):
    """
    A thread responsible for executing a single script within a DeviceThread.
    It gathers data from the parent device and its neighbors at a specific location,
    runs the assigned script, and then updates the data on the involved devices.
    """
    def __init__(self, script, location, neighbours, device):
        """
        Initializes a SlaveThread.

        Args:
            script (Script): The script object to be executed.
            location (int): The logical location (index) associated with this script.
            neighbours (List[Device]): A list of neighboring Device instances
                                       whose data might be relevant.
            device (Device): The parent Device instance that spawned this thread.
        """
        Thread.__init__(self, name="Slave Thread of Device %d" % device.device_id)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        The main execution method for SlaveThread.
        It gathers data from the device and its neighbors for a specific location,
        executes the script, and then updates the data on the device and its neighbors.
        """
        device = self.device
        script = self.script
        location = self.location
        neighbours = self.neighbours
        
        # Block Logic: Gathers input data for the script.
        data = device.get_data(location)
        input_data = []
        this_lock = device.lock_location[location] # Acquire the lock for this specific location.

        if data is not None:
            input_data.append(data) # Add data from the current device.

        # Block Logic: Acquires the lock for the current location to ensure exclusive
        # access to the data at this location across all devices for script processing.
        with this_lock: 
            # Block Logic: Gathers data from neighboring devices for the same location.
            for neighbour in neighbours:
                temp = neighbour.get_data(location) 

                if temp is not None:
                    input_data.append(temp)

            # Conditional Logic: If there is any data to process, run the script.
            if input_data != []: 
                result = script.run(input_data) # Execute the script.

                # Block Logic: Updates the data on neighboring devices with the script's result.
                for neighbour in neighbours:
                    neighbour.set_data(location, result) 

                # Block Logic: Updates the data on the current device with the script's result.
                device.set_data(location, result) 


class ReusableBarrierSem():
    """
    Implements a reusable barrier synchronization mechanism using semaphores.
    This barrier allows a fixed number of threads (`num_threads`) to wait for each other
    at a synchronization point, and then allows them to proceed. It is "reusable"
    because it can be used multiple times without reinitialization.
    It uses a two-phase approach to prevent threads from "slipping" through the barrier
    if they arrive too early for the next cycle.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrierSem.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any of them can proceed.
        """
        self.num_threads = num_threads
        # Counter for threads in phase 1, initialized to total threads.
        self.count_threads1 = self.num_threads

        # Counter for threads in phase 2, initialized to total threads.
        self.count_threads2 = self.num_threads
        
        # Lock to protect access to the thread counters.
        self.counter_lock = Lock()
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
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        The first phase of the barrier. Threads decrement a counter and the last thread
        to arrive releases all waiting threads for phase 1.
        """
        with self.counter_lock: # Protect access to the counter.
            self.count_threads1 -= 1
            # Conditional Logic: If this is the last thread to arrive at phase 1.
            if self.count_threads1 == 0:
                # Release all `num_threads` from the first semaphore.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            # Reset the counter for phase 2.
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire() # Block until released by the last thread in phase 1.
         
    def phase2(self):
        """
        The second phase of the barrier. Threads decrement a counter and the last thread
        to arrive releases all waiting threads for phase 2. This phase ensures reusability.
        """
        with self.counter_lock: # Protect access to the counter.
            self.count_threads2 -= 1
            # Conditional Logic: If this is the last thread to arrive at phase 2.
            if self.count_threads2 == 0:
                # Release all `num_threads` from the second semaphore.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            # Reset the counter for phase 1 for the next barrier cycle.
            self.count_threads1 = self.num_threads
         
        self.threads_sem2.acquire() # Block until released by the last thread in phase 2.
