"""
This module defines the core components for a distributed device simulation.

It implements a system where each `Device` acts as a node, managing sensor data,
executing scripts, and coordinating with a supervisor and other devices.
The architecture leverages dynamically created `ScriptWorker` threads for concurrent
script execution, and employs `Lock`s and a custom `ReusableBarrier` (Semaphore/Lock-based)
to ensure thread safety and synchronized operations across the distributed system.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    A reusable barrier implementation using `threading.Semaphore` and `threading.Lock`.

    This barrier allows a specified number of threads to synchronize,
    waiting for all threads to reach the barrier before any can proceed.
    It uses a two-phase approach to ensure reusability.
    """
    
    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can proceed.
        """
        self.num_threads = num_threads
        # Counters for the two phases of the barrier. Using a list to make it mutable for pass-by-reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # Lock to protect the counters from race conditions.
        self.count_lock = Lock()
        # Semaphores to block and unblock threads in each phase.
        self.threads_sem1 = Semaphore(0) # Initially blocked (no permits).
        self.threads_sem2 = Semaphore(0) # Initially blocked (no permits).

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this barrier.
        This is a two-phase barrier to ensure proper reusability.
        """
        self.phase(self.count_threads1, self.threads_sem1) # Execute the first phase of synchronization.
        self.phase(self.count_threads2, self.threads_sem2) # Execute the second phase of synchronization.

    def phase(self, count_threads, threads_sem):
        """
        Implements a single phase of the barrier synchronization.

        Threads decrement a shared counter. The last thread to reach zero releases
        all waiting threads on the given semaphore and resets the counter.
        All threads then acquire a permit from the semaphore.

        Args:
            count_threads (list): A mutable list containing the integer counter for this phase.
            threads_sem (Semaphore): The semaphore associated with this phase.
        """
        with self.count_lock: # Protect access to the shared counter.
            count_threads[0] -= 1 # Decrement the counter.
            if count_threads[0] == 0: # If this is the last thread to reach this phase.
                n_threads = self.num_threads # Get the total number of threads.
                while n_threads > 0:
                    threads_sem.release() # Release a permit for each thread.
                    n_threads -= 1
                count_threads[0] = self.num_threads # Reset counter for next use.
        threads_sem.acquire() # Block until a permit is available (released by the last thread).


class Device(object):
    """
    Represents a single device (node) in a distributed simulation environment.

    Each device manages its own sensor data, processes assigned scripts, and
    coordinates with a central supervisor and other devices for synchronized operations.
    It uses a dedicated `DeviceThread` for its control logic, and dynamically creates
    `ScriptWorker` threads for script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique integer identifier for this device.
            sensor_data (dict): A dictionary mapping locations (str) to sensor data (object).
            supervisor (object): An object providing central coordination, typically used
                                 to retrieve information about other devices or global state.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Signals when a script has been assigned.
        self.scripts = []               # List to store (script, location) tuples assigned to this device.
        self.timepoint_done = Event()   # Signals that scripts for a timepoint are fully assigned.
        
        # The DeviceThread manages the core logic of this device.
        self.thread = DeviceThread(self)
        self.devices = [] # List to store references to all devices in the system.
        self.barrier = None # ReusableBarrier for global synchronization, initialized in `setup_devices`.
        self.workers = [] # List to store references to dynamically created `ScriptWorker` threads.
        
        # A dictionary to hold location-specific Locks (`loc_barrier` as named in original code).
        # Initialized for a fixed range of keys (0-59), with `None` as default.
        keys = range(60)
        self.loc_barrier = {key: None for key in keys}
        self.thread.start() # Start the main device control thread.

    def __str__(self):
        """
        Returns a human-readable string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of the shared `ReusableBarrier` across all devices.

        If the barrier has not been initialized yet, it creates a new `ReusableBarrier`
        and distributes its reference to all other devices. Also populates the `self.devices`
        list with references to all devices.

        Args:
            devices (list): A list of all Device instances in the distributed system.
        """
        # Block Logic: Initialize and distribute the shared `ReusableBarrier` if not already done.
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices)) # Create a new barrier instance.
            self.barrier = barrier # Assign to self.
            for device in devices:
                if device.barrier is None: # Distribute to other devices that haven't received it.
                    device.barrier = barrier

        # Populates the internal `self.devices` list with references to all devices.
        for device in devices:
            if device is not None: # Ensure the device object is valid.
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device or signals a timepoint completion.

        If a script object is provided, it's added to the device's internal `scripts` list.
        It also ensures a `Lock` exists for the given `location` in `loc_barrier`,
        creating one if necessary and propagating it among devices.
        If `script` is `None`, it signals that no more scripts are expected for the current
        timepoint, setting the `timepoint_done` event.

        Args:
            script (object): The script object to be executed, or `None` to signal timepoint completion.
            location (int): The identifier for the data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
            
            # Block Logic: Ensure a Lock exists for the location in `loc_barrier`.
            # If the current device doesn't have a lock for this location, it tries
            # to find one from other devices or creates a new one.
            if self.loc_barrier[location] is None:
                found_lock = False
                for device in self.devices: # Search among all devices for an existing lock for this location.
                    if device.loc_barrier[location] is not None:
                        self.loc_barrier[location] = device.loc_barrier[location] # Use the existing lock.
                        found_lock = True
                        break
                if not found_lock: # If no existing lock found, create a new one.
                    self.loc_barrier[location] = Lock() # Create a new Lock for this location.
            
            self.script_received.set() # Signal that a script has been assigned.
        else:
            # If script is None, it signifies the end of script assignment for this timepoint.
            self.timepoint_done.set() # Signal that this timepoint is done receiving scripts.

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from this device's local storage.

        Args:
            location (str): The identifier of the data location.

        Returns:
            object: The sensor data at the given location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location on this device's local storage.

        Args:
            location (str): The identifier of the data location.
            data (object): The new data value to set for the location.
        """
        # Data is only updated if the location already exists in sensor_data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates a graceful shutdown of the device's main `DeviceThread`.
        """
        self.thread.join() # Wait for the DeviceThread to complete its execution.


class ScriptWorker(Thread):
    """
    A worker thread dynamically created to execute a single script for a `Device`.

    This thread is responsible for collecting data from its device and neighbors,
    executing a specific script, and then updating the data on its device and neighbors.
    It uses shared `Lock`s for managing concurrent data updates at specific locations.
    """
    
    def __init__(self, device, neighbours, script, location):
        """
        Initializes a `ScriptWorker`.

        Args:
            device (Device): The `Device` instance this worker belongs to.
            neighbours (list): A list of neighboring `Device` instances.
            script (object): The script object to be executed.
            location (int): The identifier for the data location the script operates on.
        """
        Thread.__init__(self, name="Script Worker for Device %d, Loc %d" % (device.device_id, location))
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        The main execution loop for the `ScriptWorker` thread.

        This loop:
        1. Acquires the location-specific lock from the shared `device.loc_barrier` dictionary.
        2. Collects data from its own device and neighboring devices.
        3. Executes the assigned script with the collected data.
        4. Updates data on its own device and neighboring devices with the script's result.
        5. Releases the location-specific lock.
        """
        # Critical Section: Acquire the lock for the specific location before accessing or modifying data.
        self.device.loc_barrier[self.location].acquire()
        
        script_data = [] # List to store collected data.
        
        # Block Logic: Gather data from neighboring devices.
        for device_neigh in self.neighbours: # Iterate through neighbors to collect data.
            # No explicit lock around get_data for neighbors, assuming `loc_barrier` on current location
            # is sufficient or data access pattern is designed to avoid conflicts.
            data = device_neigh.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Gather data from the current device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []: # Only run script if data was collected.
            result = self.script.run(script_data) # Execute the script.
            
            # Block Logic: Update data on neighboring devices.
            for device_neigh in self.neighbours: # Iterate through neighbors to update data.
                # No explicit lock around set_data for neighbors, similar to get_data.
                device_neigh.set_data(self.location, result)
            
            # Block Logic: Update data on the current device.
            self.device.set_data(self.location, result)
        
        # Release the lock for the specific location.
        self.device.loc_barrier[self.location].release()


class DeviceThread(Thread):
    """
    The main thread for a `Device`, responsible for orchestrating timepoint processing.

    It manages synchronization using a global `ReusableBarrier`, fetches neighbor
    information from the supervisor, and dynamically creates and manages `ScriptWorker` threads
    for concurrent script execution within each timepoint.
    """

    def __init__(self, device):
        """
        Initializes the `DeviceThread`.

        Args:
            device (Device): The `Device` instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the `DeviceThread`.

        This loop continuously:
        1. Retrieves neighbor information from the supervisor. If `None` is returned,
           it signifies a system-wide shutdown and the thread terminates.
        2. Waits for scripts to be assigned via `timepoint_done` event.
        3. For each assigned script, it dynamically creates a `ScriptWorker`, starts it,
           and then stores it in `self.device.workers`.
        4. Waits for all `ScriptWorker`s created in this timepoint to complete.
        5. Clears the list of workers.
        6. Clears the `timepoint_done` event.
        7. Synchronizes with all other devices using the global `ReusableBarrier`.
        """
        # No pre-allocation needed for worker threads list; it's managed by `self.device.workers`.
        while True:
            # Block Logic: Retrieve current neighbours and handle system shutdown.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Check for a shutdown signal from the supervisor.
                break # Exit the timepoint processing loop.

            # Synchronization Point: Wait for all scripts for the current timepoint to be assigned.
            self.device.timepoint_done.wait() # Blocks until `assign_script` (called externally) sets this event.

            # Block Logic: Create and manage `ScriptWorker` threads for each assigned script.
            for (script, location) in self.device.scripts:
                # Create a new ScriptWorker for each script.
                worker = ScriptWorker(self.device, neighbours, script, location)
                self.device.workers.append(worker) # Add worker to list.

            # Start all worker threads.
            for worker in self.device.workers:
                worker.start()

            # Wait for all dynamically created worker threads to complete their tasks for the current timepoint.
            for worker in self.device.workers:
                worker.join() # Block until the worker thread finishes.

            # Clear the list of workers and scripts for the next timepoint.
            self.device.workers = []
            self.device.scripts = []
            self.device.timepoint_done.clear() # Clear the event for the next timepoint.
            
            # Global Synchronization Point: Wait for all devices to reach this barrier
            # before advancing to the next timepoint.
            self.device.barrier.wait()
