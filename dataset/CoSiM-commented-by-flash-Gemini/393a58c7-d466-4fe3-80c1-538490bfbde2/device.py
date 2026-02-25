


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a simulated device within a multi-device system, capable of
    acting as a master or slave for shared resource management.

    Each device manages its sensor data, handles script assignments,
    and orchestrates script execution via dedicated worker threads,
    synchronizing operations through shared barriers and data-specific locks.
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
        # Event to signal when shared data locks have been initialized by the master.
        self.are_locks_ready = Event() 
        # Identifier of the master device; None if this device is the master.
        self.master_id = None
        # Boolean flag indicating if this device is the master.
        self.is_master = True 
        # Reference to the shared reusable barrier for inter-device synchronization.
        self.barrier = None 
        # List to store references to all devices in the system for easier access.
        self.stored_devices = [] 
        # Array of Locks for protecting access to sensor data at different locations.
        self.data_lock = [None] * 100 
        # Event used by the master device to signal when its setup is complete.
        self.master_barrier = Event() 
        # A simple lock for protecting individual operations, e.g., set_data.
        self.lock = Lock() 
        # List to keep track of currently running ExecutorThread instances.
        self.started_threads = [] 
        # Unique identifier for this device.
        self.device_id = device_id
        # Dictionary storing this device's sensor data.
        self.sensor_data = sensor_data
        # Reference to the supervisor object.
        self.supervisor = supervisor
        # Event to signal when new scripts have been received for the current timepoint.
        self.script_received = Event()
        # List of scripts assigned to this device for current processing round.
        self.scripts = []
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
        Configures the device, establishing a master-slave relationship and
        distributing shared resources like the synchronization barrier and
        data locks.

        Args:
            devices (list): A list of all `Device` instances in the system.
        """
        # Determine if this device should be a slave by checking if any other device
        # has already identified a master.
        for device in devices:
            if device is not None and device.master_id is not None:
                self.master_id = device.master_id
                self.is_master = False
                break

        # If this device is the master, it initializes shared resources.
        if self.is_master is True:
            # Initialize the shared reusable barrier with the total number of devices.
            self.barrier = ReusableBarrierSem(len(devices))
            # Set this device as the master.
            self.master_id = self.device_id
            # Initialize data locks for all possible locations.
            for i in range(100):
                self.data_lock[i] = Lock()
            # Signal that the data locks are ready for use by other devices.
            self.are_locks_ready.set()
            # Signal that the master's setup is complete.
            self.master_barrier.set()
            # Distribute the shared barrier to all other devices.
            for device in devices:
                if device is not None:
                    device.barrier = self.barrier
                    self.stored_devices.append(device)
        else: 
            # If this device is a slave, it waits for the master to initialize resources.
            for device in devices:
                if device is not None:
                    if device.device_id == self.master_id:
                        # Wait for the master to complete its initial setup.
                        device.master_barrier.wait()
                        # If barrier hasn't been set, retrieve it from the master.
                        if self.barrier is None:
                            self.barrier = device.barrier
                    self.stored_devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device.

        If a script is provided, it's added to the device's list of scripts.
        It also handles the propagation of the `data_lock` array from the master
        device to this slave device. If `script` is None, it signals that
        script assignment for the current timepoint is complete.

        Args:
            script (object or None): The script object to assign, or None to signal
                                     the end of script assignments for a timepoint.
            location (int): The identifier for the location associated with the script.
        """
        if script is not None:
            # Add the script and its location to the device's scripts list.
            self.scripts.append((script, location))
            # Wait for the master device to indicate that data locks are ready.
            for device in self.stored_devices:
                if device.device_id == self.master_id:
                    device.are_locks_ready.wait()
            # Retrieve the shared data_lock array from the master device.
            for device in self.stored_devices:
                if device.device_id == self.master_id:
                    self.data_lock = device.data_lock
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

        This method uses an internal lock to ensure thread-safe updates to the
        sensor data dictionary.

        Args:
            location (int): The identifier for the location where the data should be set.
            data (Any): The new data to set for the location.
        """
        # Acquire a lock to protect access to the sensor_data dictionary.
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        # Release the lock.
        self.lock.release()

    def shutdown(self):
        """
        Shuts down the device by joining its associated thread.
        This ensures that the device's main thread completes its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the execution rounds for a Device, orchestrating `ExecutorThread`s.

    This thread is responsible for continuously fetching neighbor information,
    waiting for scripts to be assigned for a timepoint, dispatching these scripts
    to individual `ExecutorThread` instances for parallel execution, waiting
    for their completion, and then synchronizing with other DeviceThreads
    using a shared barrier.
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
        1. Retrieves up-to-date neighbor information from the supervisor.
        2. If no neighbors are returned (e.g., simulation end), the loop breaks.
        3. Waits until all scripts for the current timepoint are assigned and ready.
        4. Creates and starts an `ExecutorThread` for each assigned script,
           allowing concurrent processing.
        5. Waits for all `ExecutorThread` instances to complete their tasks.
        6. Clears the list of started `ExecutorThread`s for the next round.
        7. Clears the `timepoint_done` event to prepare for the next cycle.
        8. Synchronizes with other DeviceThreads using the shared `barrier`.
        """
        while True:
            # Retrieve updated neighbor information from the supervisor for the current round.
            neighbours = self.device.supervisor.get_neighbours()
            # If supervisor returns None, it signals the simulation to terminate.
            if neighbours is None:
                break

            # Wait until the current set of scripts for this timepoint has been assigned.
            self.device.timepoint_done.wait()


            # For each assigned script, create and start an ExecutorThread for concurrent execution.
            for (script, location) in self.device.scripts:
                executor = ExecutorThread(self.device, script, neighbours, location)
                self.device.started_threads.append(executor)
                executor.start()

            # Wait for all ExecutorThread instances to complete their execution.
            for executor in self.device.started_threads:
                executor.join()

            # Clear the list of started ExecutorThread instances for the next timepoint.
            del self.device.started_threads[:]
            # Clear the event, indicating that scripts for this round have been processed.
            self.device.timepoint_done.clear()
            # Wait at the shared barrier to synchronize with all other devices.
            self.device.barrier.wait()


class ExecutorThread(Thread):
    """
    A worker thread responsible for executing a single script for a device
    at a specific location.

    Each `ExecutorThread` instance handles a script, collects relevant data
    from the device and its neighbors, executes the script, and updates
    the sensor data. It uses a location-specific lock to ensure data consistency.
    """

    def __init__(self, device, script, neighbours, location):
        """
        Initializes a new ExecutorThread instance.

        Args:
            device (Device): The Device object associated with this thread.
            script (object): The script object to execute.
            neighbours (list): A list of neighboring Device objects for data collection.
            location (int): The integer identifier of the location to process.
        """
        Thread.__init__(self, name="Executor Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.location = location

    def run(self):
        """
        The main execution method for ExecutorThread.

        It performs the following steps:
        1. Acquires a lock for the specific location to ensure exclusive access
           while processing.
        2. Handles the case where neighbors might be None (e.g., at simulation end).
        3. Collects sensor data from the current device and its neighbors
           for the given location.
        4. If collected data is available, it executes the assigned script.
        5. Updates the sensor data in both the current device and its neighbors
           with the script's result.
        6. Finally, it releases the location lock.
        """
        # Acquire the lock for the current location to prevent race conditions during data access.
        self.device.data_lock[self.location].acquire()

        # If neighbours are None, it implies the simulation is ending or already ended.
        if self.neighbours is None:
            # Release the lock before returning to avoid deadlocks.
            self.device.data_lock[self.location].release()
            return

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
            
            # Update the data in neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.location, result)
            # Update the data in the current device with the script's result.
            self.device.set_data(self.location, result)

        # Release the lock for the current location.
        self.device.data_lock[self.location].release()
