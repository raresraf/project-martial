


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier synchronization primitive for coordinating multiple threads.
    This barrier allows a fixed number of threads to wait until all threads
    have reached a specific point, and then resets itself for reuse.
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
        self.count_threads1 = [self.num_threads]
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
        called this method. This method involves two phases to ensure the
        barrier can be reused.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Manages a single phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current count of threads
                                  remaining in this phase. (Uses a list to allow
                                  modification within the `with` statement).
            threads_sem (Semaphore): The semaphore used to release threads
                                     once the count reaches zero.
        """
        with self.count_lock:
            # Decrement the count of threads waiting in this phase.
            count_threads[0] -= 1
            # If this is the last thread to arrive, release all waiting threads.
            if count_threads[0] == 0:            
                # Release all threads by calling release() num_threads times.
                for i in range(self.num_threads):
                    threads_sem.release()        
                # Reset the thread count for the next use of the barrier.
                count_threads[0] = self.num_threads  
        # Acquire the semaphore, effectively waiting until all threads are released.
        threads_sem.acquire()


class Device(object):
    """
    Represents a simulated device within a multi-device system.

    Each device has unique sensor data, can execute scripts, and interacts
    with a supervisor and other devices through synchronization mechanisms
    like a reusable barrier and locks for managing specific locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor readings,
                                 where keys are locations and values are data.
            supervisor (Supervisor): A reference to the supervisor object
                                     that manages device interactions.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been received.
        self.script_received = Event()
        # List to store scripts assigned to this device, each with its associated location.
        self.scripts = []
        # Event to signal that processing for the current timepoint is done.
        self.timepoint_done = Event()

        # The thread dedicated to running this device's logic.
        self.thread = DeviceThread(self)
        self.thread.start()
        # List of all devices in the system, populated during setup.
        self.devices = []
        # The reusable barrier for synchronizing all devices.
        self.reusable_barrier = None
        # List to hold references to MyThread instances for current timepoint.
        self.thread_list = []
        # Array to store locks for specific locations (shared resources).
        # Initialized with None, locks are created on first access per location.
        self.location_lock = [None] * 99

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources (like the reusable barrier) for all devices.
        If the barrier is not yet initialized for this device, it creates a new one
        and shares it with all other devices.

        Args:
            devices (list): A list of all Device instances in the system.
        """
        # Check if the reusable barrier has been initialized for this device.
        if self.reusable_barrier is None:
            # If not, create a new barrier with the total number of devices.
            barrier = ReusableBarrier(len(devices))
            self.reusable_barrier = barrier
            # Share the newly created barrier with all other devices.
            for device in devices:
                if device.reusable_barrier is None:
                    device.reusable_barrier = barrier

        # Populate the internal list of devices with all devices in the system.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location and manages location locks.

        If `script` is None, it signals that no more scripts are to be executed
        for the current timepoint, and the `timepoint_done` event is set.

        Args:
            script (Script or None): The script object to assign, or None to signal
                                     the end of scripts for a timepoint.
            location (int): The integer identifier for the location associated with the script.
        """
        is_location_locked = 0
        if script is not None:
            # Add the script and its location to the device's script list.
            self.scripts.append((script, location))
            # If no lock exists for this location, initialize it.
            if self.location_lock[location] is None:
                # Check if any other device already has a lock for this location.
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        # If found, share that existing lock.
                        self.location_lock[location] = device .location_lock[location]
                        is_location_locked = 1
                        break
                # If no existing lock was found, create a new one for this location.
                if is_location_locked == 0:
                    self.location_lock[location] = Lock()
            # Signal that a script has been received.
            self.script_received.set()

        else:
            # If script is None, signal that all scripts for this timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The integer identifier for the location for which to retrieve data.

        Returns:
            Any: The sensor data if the location exists in sensor_data, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (int): The integer identifier for the location where the data should be set.
            data (Any): The new data to set for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its associated thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the execution of scripts on a Device within a simulated environment.

    This thread continually retrieves neighbor information from a supervisor,
    waits for scripts to be assigned for a timepoint, dispatches these scripts
    to individual `MyThread` instances for parallel execution, and then
    synchronizes with other device threads using a reusable barrier.
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

        It continuously performs the following steps:
        1. Retrieves information about neighboring devices from the supervisor.
        2. If there are no neighbors (indicating shutdown), breaks the loop.
        3. Waits until all scripts for the current timepoint are assigned.
        4. Creates and starts `MyThread` instances for each assigned script,
           allowing parallel processing of scripts for different locations.
        5. Waits for all `MyThread` instances to complete their execution.
        6. Clears the list of `MyThread` instances.
        7. Clears the `timepoint_done` event to prepare for the next cycle.
        8. Waits at the reusable barrier to synchronize with other device threads.
        """
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # If no neighbors are returned (e.g., simulation end), break the loop.
                break

            # Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # For each assigned script, create and start a MyThread for concurrent execution.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.thread_list.append(thread)

            # Start all MyThread instances.
            for thread in self.device.thread_list:
                thread.start()

            # Wait for all MyThread instances to complete their execution.
            for thread in self.device.thread_list:
                thread.join()

            # Clear the list of MyThread instances for the next timepoint.
            self.device.thread_list = []

            # Clear the event to prepare for the next timepoint's scripts.
            self.device.timepoint_done.clear()
            # Wait at the reusable barrier to synchronize with other device threads.
            self.device.reusable_barrier.wait()

class MyThread(Thread):
    """
    A worker thread responsible for executing a specific script for a device
    at a given location, collecting data from neighbors, and updating sensor data.
    Each instance of MyThread processes a single script-location pair.
    """

    def __init__(self, device, location, script, neighbours):
        """
        Initializes a new MyThread instance.

        Args:
            device (Device): The Device object associated with this thread.
            location (int): The integer identifier of the location to process.
            script (Script): The script object to execute.
            neighbours (list): A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        The main execution method for MyThread.

        It acquires a lock for the specific location to ensure exclusive access
        while processing. It collects sensor data from the current device and
        its neighbors for the given location, executes the assigned script with
        this data, and then updates the sensor data in both the current device
        and its neighbors with the script's result. Finally, it releases the
        location lock.
        """
        # Acquire the lock for the current location to prevent race conditions.
        self.device.location_lock[self.location].acquire()
        script_data = []
        
        # Collect data from neighboring devices for the current location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Collect data from the current device for the current location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Execute the script with the collected data.
            result = self.script.run(script_data)

            # Update the data in neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            # Update the data in the current device.
            self.device.set_data(self.location, result)

        # Release the lock for the current location.
        self.device.location_lock[self.location].release()
