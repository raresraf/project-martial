


from threading import Event, Semaphore, Lock, Thread



class ReusableBarrierSem(object):
    """
    A reusable barrier synchronization primitive that uses semaphores to coordinate multiple threads.
    It ensures that a fixed number of threads all reach a certain point before any can proceed,
    and can then be reset for subsequent synchronization points. This implementation uses a
    two-phase approach to allow for barrier reuse.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrierSem with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that must reach the
                                barrier before any can proceed.
        """
        self.num_threads = num_threads
        # Counter for the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # Counter for the second phase of the barrier.
        self.count_threads2 = self.num_threads
        
        # A lock to protect access to the thread counters.
        self.counter_lock = Lock()
        
        # Semaphore for releasing threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        
        # Semaphore for releasing threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all other threads have also
        called this method. This method orchestrates the two phases of the barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        Manages the first phase of the barrier synchronization.
        Threads decrement a shared counter and the last thread releases all others
        through a semaphore.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # If this is the last thread, release all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the counter for the next use of phase1.
                self.count_threads1 = self.num_threads
        # Acquire the semaphore, effectively waiting until all threads are released in this phase.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        Manages the second phase of the barrier synchronization.
        Similar to phase1, but uses a separate counter and semaphore for reuse.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # If this is the last thread, release all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the counter for the next use of phase2.
                self.count_threads2 = self.num_threads
        # Acquire the semaphore, effectively waiting until all threads are released in this phase.
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a simulated device within a multi-device system.

    Each device manages its own sensor data, processes assigned scripts,
    and coordinates with other devices through a shared barrier and
    a dedicated lock for thread-safe data access during updates.
    Scripts are executed concurrently by `MyScriptThread` instances.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor readings,
                                 where keys are locations and values are data.
            supervisor (Supervisor): A reference to the supervisor object
                                     that manages overall simulation and device interactions.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that scripts have been received for the current timepoint.
        self.script_received = Event()
        # List of scripts assigned to this device for current processing round.
        self.scripts = []
        
        # A lock to protect access to the device's sensor_data during updates.
        self.my_lock = Lock()
        # Reference to the shared reusable barrier for inter-device synchronization.
        # Initialized with 0, its actual size will be set in setup_devices.
        self.barrier = ReusableBarrierSem(0)
        # Event to signal that all scripts for the current timepoint are ready.
        self.timepoint_done = Event()
        # The dedicated thread for this device's main operational logic.
        self.thread = DeviceThread(self)
        # Start the device's operational thread upon initialization.
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
        Configures shared resources, specifically the synchronization barrier,
        for all devices in the system.

        If this device is device_id 0, it initializes the `ReusableBarrierSem`
        with the total number of devices. Other devices receive a reference
        to this shared barrier from device 0.

        Args:
            devices (list): A list of all `Device` instances in the system.
        """
        # The first device (device_id 0) is responsible for initializing the shared barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            # Other devices obtain a reference to the barrier initialized by device 0.
            # This relies on device 0's barrier being set up before other devices access it.
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device.

        If a script is provided, it's added to the device's list of scripts.
        If `script` is None, it signals that script assignment for the current
        timepoint is complete, and both `script_received` and `timepoint_done`
        events are set.

        Args:
            script (object or None): The script object to assign, or None to signal
                                     the end of script assignments for a timepoint.
            location (int): The identifier for the location associated with the script.
        """
        if script is not None:
            # Add the script and its location to the device's scripts list.
            self.scripts.append((script, location))
        else:
            # If script is None, signal that all scripts for this timepoint are assigned.
            self.script_received.set()
            # Also signal that this timepoint is done receiving scripts.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Note: Access to sensor data should be protected by `my_lock` during updates
        in `MyScriptThread`, but read operations here are not explicitly locked,
        implying a design choice or assumption about data consistency during reads.

        Args:
            location (int): The identifier for the location for which to retrieve data.

        Returns:
            Any: The sensor data if the location exists in sensor_data, otherwise None.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Note: This method assumes the caller (e.g., `MyScriptThread`) will acquire
        and release `self.my_lock` around calls to this method to ensure thread safety.

        Args:
            location (int): The identifier for the location where the data should be set.
            data (Any): The new data to set for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its associated main thread.
        This ensures that the device's main thread completes its execution.
        """
        self.thread.join()



class MyScriptThread(Thread):
    """
    A worker thread responsible for executing a single script for a device
    at a specific location.

    This thread collects relevant data from the device and its neighbors,
    executes the script, and updates the sensor data. It explicitly
    acquires and releases the `device.my_lock` around data updates to
    ensure thread safety.
    """
    

    def __init__(self, script, location, device, neighbours):
        """
        Initializes a new MyScriptThread instance.

        Args:
            script (object): The script object to execute.
            location (int): The integer identifier of the location to process.
            device (Device): The Device object associated with this thread.
            neighbours (list): A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        The main execution method for MyScriptThread.

        It performs the following steps:
        1. Collects sensor data from the current device and its neighbors
           for the given location.
        2. If collected data is available, it executes the assigned script.
        3. Updates the sensor data in both the current device and its neighbors
           with the script's result, ensuring thread safety by acquiring
           `device.my_lock` before updating and releasing it afterwards.
        """
        script_data = [] # List to store data collected for the script.

        # Collect data from neighboring devices for the current location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Collect data from the current device itself for the current location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)


        # If any data was collected, execute the script.
        if script_data != []:
            # Execute the script with the collected data.
            result = self.script.run(script_data)

            # Update the data in neighboring devices.
            # Acquire device.my_lock before updating to ensure thread safety.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            # Update the data in the current device.
            # Acquire device.my_lock before updating to ensure thread safety.
            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()

class DeviceThread(Thread):
    """
    Manages the overall timestep progression and coordinates script processing
    for a Device.

    This thread continuously fetches neighbor information, waits for scripts
    to be assigned for a timepoint, dispatches these scripts to individual
    `MyScriptThread` instances for parallel execution, waits for their
    completion, and then synchronizes with other DeviceThreads using a
    shared barrier.
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
        The main execution loop for the DeviceThread.

        Each iteration represents a processing round. It performs the following steps:
        1. Retrieves updated neighbor information from the supervisor.
        2. If no neighbors are returned (e.g., simulation end), the loop breaks.
        3. Synchronizes with other DeviceThreads using the shared `barrier`.
        4. Waits until scripts for the current timepoint have been assigned.
        5. Creates and starts a `MyScriptThread` for each assigned script,
           allowing concurrent processing.
        6. Waits for all `MyScriptThread` instances to complete their tasks.
        7. Waits for `timepoint_done` (signaled by `assign_script` when all scripts are handled).
        8. Synchronizes again with the shared `barrier`.
        9. Clears the `script_received` event for the next round.
        """
        while True:
            # Retrieve updated neighbor information from the supervisor for the current round.
            neighbours = self.device.supervisor.get_neighbours()
            # If supervisor returns None, it signals the simulation to terminate.
            if neighbours is None:
                break;
            
            # Wait at the shared barrier to synchronize with all other devices.
            self.device.barrier.wait()

            # Wait until scripts for the current timepoint have been assigned.
            self.device.script_received.wait()
            script_threads = [] # List to hold references to active script threads.
            
            # For each assigned script, create and start a MyScriptThread for concurrent execution.
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            # Start all MyScriptThread instances.
            for thread in script_threads:
                thread.start()
            # Wait for all MyScriptThread instances to complete their execution.
            for thread in script_threads:
                thread.join()
            
            # Clear the list of script threads.
            self.device.scripts.clear()
            
            # Wait until timepoint_done is signaled, indicating that all scripts for this timepoint have been processed.
            self.device.timepoint_done.wait()
            # Synchronize again with the shared barrier (likely for final state collection or next round preparation).
            self.device.barrier.wait()
            # Clear the event, indicating that scripts for this round have been dispatched.
            self.device.script_received.clear()
