


from threading import Event, Semaphore, Lock, Thread



class ReusableBarrierSem(object):
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
        # Block Logic: Atomically decrements the counter for phase 1.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Conditional Logic: If this is the last thread to arrive at phase 1.
            if self.count_threads1 == 0:
                # Release all `num_threads` from the first semaphore.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the counter for the next use of the barrier.
                self.count_threads1 = self.num_threads
        # Block until released by the last thread in phase 1.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        The second phase of the barrier. Threads decrement a counter and the last thread
        to arrive releases all waiting threads for phase 2. This phase ensures reusability.
        """
        # Block Logic: Atomically decrements the counter for phase 2.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Conditional Logic: If this is the last thread to arrive at phase 2.
            if self.count_threads2 == 0:
                # Release all `num_threads` from the second semaphore.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the counter for the next use of the barrier.
                self.count_threads2 = self.num_threads
        # Block until released by the last thread in phase 2.
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a single computational device in a simulated distributed system.
    Each device has an ID, sensor data, and communicates with a supervisor.
    It can receive and execute scripts, synchronizing with other devices using a barrier.
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
        
        # Lock to protect access to the device's data and state.
        self.my_lock = Lock()
        # Barrier for synchronization with other devices. Initialized to 0, will be
        # properly configured by setup_devices.
        self.barrier = ReusableBarrierSem(0)
        # Event to signal when the device has completed processing for a timepoint.
        self.timepoint_done = Event()
        # The main thread for this device, which handles its operational lifecycle.
        self.thread = DeviceThread(self)
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
        Configures the barrier for inter-device synchronization.
        Only device with ID 0 initializes the barrier with the total number of devices,
        and other devices get a reference to this shared barrier.

        Args:
            devices (List[Device]): A list of all Device instances in the system.
        """
        # Block Logic: Only the device with ID 0 initializes the barrier.
        # This ensures a single shared barrier instance for all devices.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            # Other devices receive a reference to the barrier initialized by Device 0.
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a specific location.
        If script is None, it signals that no more scripts will be assigned for the current
        timepoint, and the device should proceed.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (str): The logical location associated with the script's execution.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signals that all scripts for the current timepoint have been assigned.
            self.script_received.set()
            # Also signals that this device is done with its timepoint, perhaps no scripts to run.
            self.timepoint_done.set()


    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (str): The key for the sensor data.

        Returns:
            Any: The data associated with the location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (str): The key for the sensor data.
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



class MyScriptThread(Thread):
    """
    A thread dedicated to executing a single script for a specific device.
    It collects relevant data from the device itself and its neighbors,
    runs the script, and then updates the data on the involved devices.
    """

    def __init__(self, script, location, device, neighbours):
        """
        Initializes a MyScriptThread.

        Args:
            script (Script): The script object to be executed.
            location (str): The logical location associated with this script.
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
        The main execution method for MyScriptThread.
        It gathers data from the device and its neighbors, executes the script,
        and then updates the data on the device and its neighbors.
        """
        script_data = []

        # Block Logic: Collects data from neighboring devices for the specified location.
        # This allows scripts to operate on distributed data.
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
            # Executes the script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Updates the sensor data on neighboring devices with the script's result.
            # Each device's data update is protected by its own lock.
            for device in self.neighbours:
                device.my_lock.acquire() # Acquire lock for atomic update.
                device.set_data(self.location, result)
                device.my_lock.release() # Release lock.

            # Block Logic: Updates the sensor data on the current device with the script's result.
            self.device.my_lock.acquire() # Acquire lock for atomic update.
            self.device.set_data(self.location, result)
            self.device.my_lock.release() # Release lock.

class DeviceThread(Thread):
    """
    The main operational thread for a Device. It continuously manages the device's lifecycle,
    including synchronizing with other devices, executing assigned scripts, and
    waiting for new instructions.
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
        It repeatedly fetches neighbors, waits at a barrier, processes scripts,
        and signals completion for a timepoint. This loop simulates the
        continuous operation of a distributed device.
        """
        # Block Logic: The main operational loop for the device.
        while True:
            # Pre-condition: Device is ready for a new timepoint/iteration.
            # Post-condition: `neighbours` contains the list of current neighboring devices.
            neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (e.g., system shutdown), exit loop.
            if neighbours is None:
                break;
            
            # Block Logic: Waits at the barrier for all devices to reach this point.
            # This ensures that all devices are ready before processing the current timepoint.
            self.device.barrier.wait()
            # Block Logic: Waits for scripts to be assigned to this device for the current timepoint.
            self.device.script_received.wait()
            script_threads = [] # List to hold threads for executing scripts.
            
            # Block Logic: Creates and starts a new thread for each assigned script.
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            for thread in script_threads:
                thread.start() # Start script execution in parallel.
            # Block Logic: Waits for all script execution threads to complete.
            for thread in script_threads:
                thread.join()
            
            # Post-condition: All scripts for the current timepoint have been executed.
            # Signals that this device has completed processing for the current timepoint.
            self.device.timepoint_done.set()
            # Waits at the barrier again to ensure all devices have completed their timepoint processing.
            self.device.barrier.wait()
            # Resets the script_received event, preparing for the next set of script assignments.
            self.device.script_received.clear()

