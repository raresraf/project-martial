


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier synchronization primitive for coordinating multiple threads.
    This barrier allows a fixed number of threads to wait until all threads
    have reached a specific point, and then resets itself for reuse.
    It utilizes a two-phase semaphore-based mechanism to ensure correct
    operation across multiple waits.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that must reach the
                                barrier before any can proceed.
        """
        self.num_threads = num_threads
        # Tracks the number of threads waiting in the first phase of the barrier.
        self.count_threads1 = [self.num_threads] # Using a list to make it mutable within closures/nested scopes.
        # Tracks the number of threads waiting in the second phase of the barrier.
        self.count_threads2 = [self.num_threads]
        # A lock to protect access to the thread count.
        self.count_lock = Lock()
        # Semaphore for releasing threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for releasing threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all other threads have also
        called this method. This method orchestrates the two phases of the
        barrier to ensure proper synchronization and reusability.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Manages a single phase of the barrier synchronization.

        Threads decrement a shared counter. The last thread to reach zero
        releases all waiting threads using a semaphore, and then resets
        the counter for the next use.

        Args:
            count_threads (list): A mutable list containing the current count
                                  of threads remaining in this phase.
            threads_sem (Semaphore): The semaphore used to release threads
                                     once the count reaches zero.
        """
        with self.count_lock:
            # Decrement the count of threads waiting in this phase.
            count_threads[0] -= 1
            # If this is the last thread to arrive in this phase.
            if count_threads[0] == 0:
                # Release all waiting threads by calling release() num_threads times.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # Acquire the semaphore, effectively waiting until all threads are released by the last thread.
        threads_sem.acquire()

class Device(object):
    """
    Represents a simulated device within a multi-device system.

    Each device manages its own sensor data, processes assigned scripts,
    and coordinates with other devices through a shared barrier and
    location-specific locks for data consistency. Scripts are executed
    concurrently by `NewThread` instances.
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
        # List to store references to all devices in the system.
        self.devices = []
        # Event to signal that all scripts for the current timepoint are ready.
        self.timepoint_done = Event()

        # The dedicated thread for this device's main operational logic.
        self.thread = DeviceThread(self)
        # Reference to the shared reusable barrier for inter-device synchronization.
        self.barrier = None
        # List to hold references to active `NewThread` instances for concurrent script execution.
        self.list_thread = []
        # Array of Locks for protecting access to sensor data at different locations.
        self.location_lock = [None] * 100
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

        If the shared barrier is not yet initialized for this device, it creates
        a new `ReusableBarrier` and shares it with all other devices. It also
        populates this device's internal list of all devices.

        Args:
            devices (list): A list of all `Device` instances in the system.
        """
        # Check if the shared barrier has been initialized for this device.
        if self.barrier is None:
            # If not, create a new reusable barrier with the total number of devices.
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            # Share the newly created barrier with all other devices.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        # Populate the internal list of devices with all devices in the system.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device.

        If a script is provided, it's added to the device's list of scripts.
        It also handles the initialization or sharing of `location_lock`s for
        the given location. If `script` is None, it signals that script
        assignment for the current timepoint is complete.

        Args:
            script (object or None): The script object to assign, or None to signal
                                     the end of script assignments for a timepoint.
            location (int): The identifier for the location associated with the script.
        """
        ok = 0 # Flag to indicate if a location lock was successfully shared.

        if script is not None:
            # Add the script and its location to the device's scripts list.
            self.scripts.append((script, location))
            # If no lock exists for this location, initialize it or share an existing one.
            if self.location_lock[location] is None:
                # Iterate through all known devices to find an existing lock for this location.
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        # If found, use the existing lock.
                        self.location_lock[location] = device.location_lock[location]
                        ok = 1 # Mark that a lock was shared.
                        break
                # If no existing lock was found, create a new one for this location.
                if ok == 0:
                    self.location_lock[location] = Lock()
            # Signal that scripts have been received for this timepoint.
            self.script_received.set()
        else:
            # If script is None, signal that all scripts for this timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The identifier for the location for which to retrieve data.

        Returns:
            Any: The sensor data if the location exists in sensor_data, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

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

class NewThread(Thread):
    """
    A worker thread responsible for executing a single script for a device
    at a specific location.

    This thread acquires a location-specific lock, collects relevant data
    from the device and its neighbors, executes the script, and updates
    the sensor data. It ensures data consistency for the given location.
    """
    
    def __init__(self, device, location, script, neighbours):
        """
        Initializes a new NewThread instance.

        Args:
            device (Device): The Device object associated with this thread.
            location (int): The integer identifier of the location to process.
            script (object): The script object to execute.
            neighbours (list): A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        The main execution method for NewThread.

        It performs the following steps:
        1. Acquires a lock for the specific location to ensure exclusive access.
        2. Collects sensor data from the current device and its neighbors
           for the given location.
        3. If collected data is available, it executes the assigned script.
        4. Updates the sensor data in both the current device and its neighbors
           with the script's result.
        5. Finally, it releases the location lock.
        """
        script_data = [] # List to store data collected for the script.
        # Acquire the lock for the current location to prevent race conditions during data access.
        self.device.location_lock[self.location].acquire()
        
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
            
            # Update the data in neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.location, result)
                
            # Update the data in the current device with the script's result.
            self.device.set_data(self.location, result)
        # Release the lock for the current location.
        self.device.location_lock[self.location].release()

class DeviceThread(Thread):
    """
    Manages the overall timestep progression and coordinates script processing
    for a Device.

    This thread continuously fetches neighbor information, waits for scripts
    to be assigned for a timepoint, dispatches these scripts to individual
    `NewThread` instances for parallel execution, waits for their completion,
    and then synchronizes with other DeviceThreads using a shared barrier.
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
        3. Waits until all scripts for the current timepoint are assigned and ready.
        4. Creates and starts a `NewThread` for each assigned script,
           allowing concurrent processing.
        5. Waits for all `NewThread` instances to complete their tasks.
        6. Clears the list of started `NewThread`s for the next round.
        7. Clears the `timepoint_done` event to prepare for the next cycle.
        8. Synchronizes with other DeviceThreads using the shared `barrier`.
        """
        while True:
            # Retrieve updated neighbor information from the supervisor for the current round.
            neighbours = self.device.supervisor.get_neighbours()
            # If supervisor returns None, it signals the simulation to terminate.
            if neighbours is None:
                break

            # Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()


            # For each assigned script, create and start a NewThread for concurrent execution.
            for (script, location) in self.device.scripts:
                thread = NewThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            # Start all NewThread instances.
            for thread_elem in self.device.list_thread:
                thread_elem.start()
            # Wait for all NewThread instances to complete their execution.
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            # Clear the list of started NewThread instances for the next timepoint.
            self.device.list_thread = []

            # Clear the event, indicating that scripts for this round have been processed.
            self.device.timepoint_done.clear()
            # Wait at the shared barrier to synchronize with all other devices.
            self.device.barrier.wait()