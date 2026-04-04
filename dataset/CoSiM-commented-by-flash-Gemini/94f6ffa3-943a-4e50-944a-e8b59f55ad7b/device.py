

"""
@94f6ffa3-943a-4e50-944a-e8b59f55ad7b/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices with a master-slave synchronization model.

This module defines the core components for simulating a network of sensor devices,
each capable of executing scripts, managing local sensor data, and interacting
with a central supervisor. This version employs a master-slave pattern for
initial synchronization and uses `ReusableBarrierSem` for timepoint coordination.
Individual script executions are handled by `ExecutorThread` instances.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate with neighbors under the
guidance of a supervisor.

Classes:
- Device: Represents a single simulated sensor device.
- DeviceThread: Manages the lifecycle and operation of a Device, including coordinating
                `ExecutorThread` instances.
- ExecutorThread: A worker thread responsible for executing a single script for a specific location.

Domain: Distributed Systems Simulation, Concurrent Programming, Parallel Processing, Sensor Networks, Master-Slave Architecture.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. This version
    utilizes a master-slave model for synchronization setup, where one device
    acts as a master to initialize global resources like barriers and locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data, keyed by location.
        @param supervisor: A reference to the central supervisor managing all devices.
        """
        # Event to signal when location-specific locks (`data_lock`) are ready.
        self.are_locks_ready = Event() 
        # Identifier of the master device in the simulation.
        self.master_id = None
        # Boolean flag indicating if this device is the master.
        self.is_master = True 
        # Reference to the global synchronization barrier (ReusableBarrierSem).
        self.barrier = None 
        # List to store references to all other devices in the simulation.
        self.stored_devices = [] 
        # Array of Locks, where each index corresponds to a location, protecting concurrent access to data for that location.
        self.data_lock = [None] * 100 
        # Event used by the master to signal that its barrier has been initialized.
        self.master_barrier = Event() 
        # General-purpose lock for internal device operations.
        self.lock = Lock() 
        # List to keep track of currently running ExecutorThread instances.
        self.started_threads = [] 
        # Unique identifier for this device.
        self.device_id = device_id
        # Dictionary storing sensor data, keyed by location.
        self.sensor_data = sensor_data
        # Reference to the central supervisor.
        self.supervisor = supervisor
        # Event to signal that a new script has been assigned to this device.
        self.script_received = Event()
        # List to store assigned scripts, each being a tuple of (script, location).
        self.scripts = []
        # Event to signal that all scripts for the current timepoint have been processed.
        self.timepoint_done = Event()
        # The main thread responsible for the device's lifecycle.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Orchestrates the setup of synchronization mechanisms across all devices,
               following a master-slave pattern.

        The first device to call this method (or the one that has no other device with a master_id set)
        becomes the master. The master initializes the global barrier and location-specific locks.
        Slave devices wait for the master to complete this setup and then receive their references
        to these shared resources.

        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: Determines if this device should be the master.
        # It checks if any other device in the list has already established itself as a master.
        for device in devices:
            if device is not None and device.master_id is not None:
                self.master_id = device.master_id
                self.is_master = False
                break

        if self.is_master is True:
            # Block Logic: Master device initialization.
            # Pre-condition: This device is determined to be the master.
            # Invariant: The global barrier and location locks are initialized and ready for distribution.
            self.barrier = ReusableBarrierSem(len(devices))
            self.master_id = self.device_id
            # Initializes 100 Lock objects for `data_lock`, assuming locations are represented by integers 0-99.
            for i in range(100):
                self.data_lock[i] = Lock()
            self.are_locks_ready.set()
            self.master_barrier.set()
            # Distributes the initialized barrier to all other devices.
            for device in devices:
                if device is not None:
                    device.barrier = self.barrier
                    self.stored_devices.append(device)
        else: 
            # Block Logic: Slave device synchronization.
            # Pre-condition: This device is a slave.
            # Invariant: The slave device obtains references to the master's barrier and other shared resources.
            for device in devices:
                if device is not None:
                    if device.device_id == self.master_id:
                        device.master_barrier.wait() # Waits for the master to complete its setup.
                        if self.barrier is None:
                            # Assigns the master's barrier to this slave device.
                            self.barrier = device.barrier
                    self.stored_devices.append(device)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed for a specific location on this device.

        This method also handles the distribution of `data_lock` from the master device
        to the slave devices, ensuring all devices use the same set of locks.

        @param script: The script object to be executed, or None to signal completion.
        @param location: The location pertinent to the script execution.
        """
        if script is not None:
            # Pre-condition: `script` is not None, indicating a script needs to be assigned.
            # Invariant: The script is added to the device's script list.
            self.scripts.append((script, location))
            # Block Logic: Ensures that all devices have the `data_lock` array ready by waiting for the master.
            for device in self.stored_devices:
                if device is not None and device.device_id == self.master_id:
                    device.are_locks_ready.wait()
            # Block Logic: Slave devices copy the `data_lock` array from the master device.
            for device in self.stored_devices:
                if device is not None and device.device_id == self.master_id:
                    self.data_lock = device.data_lock
            self.script_received.set() # Signals that a script has been received.
        else:
            # Pre-condition: `script` is None, indicating no more scripts for the current timepoint.
            # Invariant: The `timepoint_done` event is set, signaling readiness to process assigned scripts.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The location for which to retrieve data.
        @return: The sensor data for the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        This method uses an internal lock (`self.lock`) to protect access to `sensor_data`
        during update operations.

        @param location: The location for which to set data.
        @param data: The new data value to be set.
        """
        # Critical Section: Acquires a lock to ensure exclusive access to `sensor_data` during modification.
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's main thread.

        Waits for the device's main thread to complete its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the execution lifecycle for a single Device instance in a dedicated thread.

    This thread is responsible for handling timepoint progression, spawning `ExecutorThread`
    instances for script execution, and synchronizing with other devices using a global barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        @param device: The Device instance that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Pre-condition: The device's synchronization mechanisms are properly set up.
        Invariant: The device continuously processes timepoints, executes assigned scripts
                   in parallel, and synchronizes with other devices until a shutdown signal is received.
        """
        while True:
            # Block Logic: Fetches the current neighbors of this device from the supervisor.
            # This allows for dynamic network topology changes between timepoints.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Pre-condition: `neighbours` is None, indicating a termination signal from the supervisor.
                # Invariant: The loop breaks, leading to thread termination.
                break

            # Block Logic: Waits for the supervisor to signal that all scripts for the current
            # timepoint have been assigned to all devices.
            self.device.timepoint_done.wait()

            # Block Logic: Spawns an `ExecutorThread` for each assigned script.
            # Each script (and its corresponding location) is handled by a dedicated thread for parallel execution.
            for (script, location) in self.device.scripts:
                executor = ExecutorThread(self.device, script, neighbours, location)
                self.device.started_threads.append(executor)
                executor.start()

            # Block Logic: Waits for all `ExecutorThread` instances to complete their execution
            # for the current timepoint.
            for executor in self.device.started_threads:
                executor.join()

            # Clears the list of started threads for the next timepoint.
            del self.device.started_threads[:]
            # Resets the timepoint_done event, preparing for the next timepoint.
            self.device.timepoint_done.clear()
            # Synchronizes with all other DeviceThreads using a global barrier.
            # This ensures all devices have finished processing their scripts before proceeding to the next timepoint.
            self.device.barrier.wait()


class ExecutorThread(Thread):
    """
    @brief A worker thread responsible for executing a single script for a specific location.

    This thread ensures that data access for its assigned location is synchronized
    using a per-location lock, preventing race conditions during script execution
    and data updates across devices.
    """

    def __init__(self, device, script, neighbours, location):
        """
        @brief Initializes the ExecutorThread.

        @param device: The parent Device instance.
        @param script: The script object to be executed.
        @param neighbours: A list of neighboring Device instances to interact with.
        @param location: The specific location for which the script will be executed.
        """
        Thread.__init__(self, name="Executor Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.location = location

    def run(self):
        """
        @brief The main execution method for the ExecutorThread.

        Pre-condition: The script, neighbors, and location are properly initialized.
        Invariant: The script is executed, and relevant data is updated while holding
                   the location-specific lock.
        """
        # Critical Section: Acquires a lock for the specific location to ensure exclusive
        # access to the data associated with this location during script execution and data updates.
        self.device.data_lock[self.location].acquire()

        if self.neighbours is None:
            # Pre-condition: `neighbours` is None, indicating no neighbors to gather data from.
            # Invariant: The method returns without executing the script, preventing potential errors.
            return

        script_data = []
        
        # Block Logic: Gathers data from neighboring devices for the current location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        # Gathers data from its own sensor_data for the current location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Executes the script with the collected data.
            result = self.script.run(script_data)
            
            # Block Logic: Updates data on neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Updates its own data with the script's result.
            self.device.set_data(self.location, result)

        # Releases the lock for the specific location, allowing other threads to access it.
        self.device.data_lock[self.location].release()
