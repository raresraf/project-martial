"""
This module defines the core components for a distributed device simulation.

It implements a system where each `Device` acts as a node, managing sensor data,
executing scripts, and coordinating with a supervisor and other devices.
The architecture leverages dynamically created `Worker` threads for concurrent
script execution, and employs `Lock`s and a custom `ReusableBarrier` (using `Condition`)
to ensure thread safety and synchronized operations across the distributed system.
"""

from threading import Event, Thread, Condition, Lock


class ReusableBarrierSem(object):
    """
    A reusable barrier implementation using Semaphores and a Lock.
    (NOTE: This class is defined but not actively used by DeviceThread/Device in this file.
           The `ReusableBarrier` class (defined at the end of the file) is used instead.)

    This barrier allows a specified number of threads to synchronize,
    waiting for all threads to reach the barrier before any can proceed.
    It's designed to be reusable for multiple synchronization points.
    """
    
    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrierSem.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can proceed.
        """
        self.num_threads = num_threads
        # Counters for the two phases of the barrier.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads

        # Lock to protect the counters from race conditions.
        self.counter_lock = Lock()
        # Semaphores to block and unblock threads in each phase.
        self.threads_sem1 = Semaphore(0) # Initially blocked (no permits).
        self.threads_sem2 = Semaphore(0) # Initially blocked (no permits).

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this barrier.
        This is a two-phase barrier.
        """
        self.phase1() # Execute the first phase of synchronization.
        self.phase2() # Execute the second phase of synchronization.

    def phase1(self):
        """
        The first phase of the barrier.

        Threads decrement a counter. The last thread to reach zero releases
        all threads waiting on `threads_sem1` and resets the counter.
        All threads then acquire a permit from `threads_sem1`.
        """
        with self.counter_lock: # Protect access to count_threads1.
            self.count_threads1 -= 1 # Decrement the counter for the first phase.
            if self.count_threads1 == 0: # If this is the last thread to reach the barrier.
                # Release all waiting threads for phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release() # Increment semaphore, allowing one thread to pass.
                self.count_threads1 = self.num_threads # Reset counter for next use.

        self.threads_sem1.acquire() # Block until a permit is available (released by the last thread).

    def phase2(self):
        """
        The second phase of the barrier, similar to phase 1.

        Ensures that all threads from phase 1 have proceeded before reusing the barrier.
        """
        with self.counter_lock: # Protect access to count_threads2.
            self.count_threads2 -= 1 # Decrement the counter for the second phase.
            if self.count_threads2 == 0: # If this is the last thread to reach the barrier.
                # Release all waiting threads for phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release() # Increment semaphore, allowing one thread to pass.
                self.count_threads2 = self.num_threads # Reset counter for next use.

        self.threads_sem2.acquire() # Block until a permit is available (released by the last thread).


class Device(object):
    """
    Represents a single device (node) in a distributed simulation environment.

    Each device manages its own sensor data, processes assigned scripts, and
    coordinates with a central supervisor and other devices for synchronized operations.
    It uses a dedicated `DeviceThread` for its control logic, and dynamically creates
    `Worker` threads for script execution.
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
        self.scripts = []               # List to store (script, location) tuples assigned to this device.
        self.scripts_done = Event()     # Signals that all scripts for a timepoint have been assigned.
        self.my_lock = Lock()           # A local Lock for coordinating updates to this device's own data.

        # Shared synchronization primitives, initialized by `setup_devices`.
        self.locations = None           # Dictionary mapping locations to Lock objects for controlling data access.
        self.barrier = None             # ReusableBarrier (Condition-based) for global synchronization across all devices.

    def __str__(self):
        """
        Returns a human-readable string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of shared synchronization primitives across all devices.

        Typically, Device with `device_id == 0` acts as a coordinator. It initializes
        the `ReusableBarrier` and a global `locations` dictionary (containing `Lock`s
        for each unique data location across all devices), then distributes references
        to these shared objects to all other devices.

        Args:
            devices (list): A list of all Device instances in the distributed system.
        """
        # Block Logic: Device 0 acts as a coordinator for setting up shared resources.
        if self.device_id is 0: # Check if this is the coordinating device (device_id 0).
            self.locations = {} # Initialize a global dictionary for location-specific Locks.
            # Initialize a reusable barrier for all devices.
            self.barrier = ReusableBarrier(len(devices));
            # Populate the global `locations` dictionary with Locks for initial sensor data.
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass # Location already has a lock.
                else:
                    self.locations[loc] = Lock() # Create a new Lock for this unique location.
        
        # Block Logic: Other devices (device_id != 0) receive references to shared resources.
        else:
            self.locations = devices[0].locations # Get the shared `locations` dictionary from device 0.
            self.barrier = devices[0].get_barrier() # Get the shared `barrier` from device 0.
            # Add Locks for any unique locations in this device's sensor data not yet in shared `locations`.
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass # Location already has a lock.
                else:
                    self.locations[loc] = Lock() # Create a new Lock for this unique location.

        # Create and start the main DeviceThread for this device.
        self.thread = DeviceThread(self, self.barrier, self.locations)
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device or signals a timepoint completion.

        If a script object is provided, it's added to the device's internal `scripts` list.
        If `script` is `None`, it signals that no more scripts are expected for the current
        timepoint, setting the `scripts_done` event.

        Args:
            script (object): The script object to be executed, or `None` to signal timepoint completion.
            location (str): The identifier for the data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
        else:
            # If script is None, it signifies the end of script assignment for this timepoint.
            self.scripts_done.set() # Signal that all scripts for the current timepoint have been assigned.

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

    def get_barrier(self):
        """
        Returns the shared `ReusableBarrier` instance.

        Returns:
            ReusableBarrier: The shared barrier object.
        """
        return self.barrier



class DeviceThread(Thread):
    """
    The main thread for a `Device`, responsible for orchestrating timepoint processing.

    It manages synchronization using a global `ReusableBarrier`, fetches neighbor
    information from the supervisor, and dynamically creates and manages `Worker` threads
    for concurrent script execution within each timepoint.
    """

    def __init__(self, device, barrier, locations):
        """
        Initializes the `DeviceThread`.

        Args:
            device (Device): The `Device` instance this thread is managing.
            barrier (ReusableBarrier): The shared `ReusableBarrier` instance for global synchronization.
            locations (dict): The shared dictionary of location-specific `Lock`s.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.locations = locations # Reference to the shared dictionary of location locks.

    def run(self):
        """
        The main execution loop for the `DeviceThread`.

        This loop continuously:
        1. Retrieves neighbor information from the supervisor. If `None` is returned,
           it signifies a system-wide shutdown and the thread terminates.
        2. Waits for scripts to be assigned via `scripts_done` event.
        3. Dynamically creates `Worker` threads for each assigned script, starts them,
           and then waits for their completion.
        4. Synchronizes with all other devices using the global `ReusableBarrier`.
        """
        # Block Logic: This list will temporarily hold the dynamically created worker threads.
        workers = [] # Renamed 'threads' to 'workers' for clarity.
        while True:
            # Block Logic: Retrieve current neighbours and handle system shutdown.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Check for a shutdown signal from the supervisor.
                break # Exit the timepoint processing loop.

            # Synchronization Point: Wait for new scripts to be assigned for this timepoint.
            self.device.scripts_done.wait()  # Blocks until `assign_script` (called externally) sets this event.
            self.device.scripts_done.clear() # Clear the event for the next timepoint.
            
            # Block Logic: Create and manage Worker threads for each assigned script.
            for (script, location) in self.device.scripts:
                # Create a new Worker thread for each script.
                w = Worker(self.device, neighbours, script, location, self.locations)
                workers.append(w) # Add worker to list.
                w.start() # Start the worker thread.

            # Wait for all dynamically created worker threads to complete their tasks for the current timepoint.
            for w in workers:
                w.join() # Block until the worker thread finishes.

            # Global Synchronization Point: Wait for all devices to reach this barrier
            # before advancing to the next timepoint.
            self.barrier.wait()
            
            # Clear the list of scripts for the next timepoint.
            self.device.scripts = []


class Worker(Thread):
    """
    A worker thread dynamically created to execute a single script for a `Device`.

    This thread is responsible for collecting data from its device and neighbors,
    executing a specific script, and then updating the data on its device and neighbors.
    It uses shared `Lock`s for managing concurrent data updates.
    """
    def __init__(self, device, neighbours, script, location, locations):
        """
        Initializes a `Worker` thread.

        Args:
            device (Device): The `Device` instance this worker belongs to.
            neighbours (list): A list of neighboring `Device` instances.
            script (object): The script object to be executed.
            location (str): The identifier for the data location the script operates on.
            locations (dict): The shared dictionary of location-specific `Lock`s.
        """
        Thread.__init__(self, name="Worker Thread %d" % device.device_id) # Name includes device ID for clarity
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.locations = locations # Reference to the shared dictionary of location locks.

    def collect_data(self, location_data):
        """
        Collects relevant data for the script from the current device and its neighbors.

        Args:
            location_data (list): A list to append the collected data to.
        """
        # Get data from this worker's own device.
        with self.device.my_lock: # Acquire local lock to protect access to its own sensor_data.
            data_self = self.device.get_data(self.location)
            if data_self is not None:
                location_data.append(data_self)
        
        # Get data from neighboring devices.
        for device_neigh in self.neighbours: # Renamed 'device' to 'device_neigh' for clarity.
            with device_neigh.my_lock: # Acquire neighbor's local lock to protect its sensor_data.
                data_neigh = device_neigh.get_data(self.location)
                if data_neigh is not None:
                    location_data.append(data_neigh)

    def update_neighbours(self, result):
        """
        Updates the data on neighboring devices with the script's result.

        Uses the shared `device.lock_neigh` to ensure thread-safe updates to neighbor data.

        Args:
            result (object): The result of the script execution.
        """
        no_neigh = len(self.neighbours)
        for i in range(no_neigh):
            # The original code's `lock_neigh` was a shared Lock on the Device object.
            # This logic assumes `max(result, value)` is the update rule.
            with self.device.lock_neigh: # Acquire shared lock for neighbor updates.
                value = self.neighbours[i].get_data(self.location) # Get current neighbor data.
                self.neighbours[i].set_data(self.location, max(result, value)) # Update neighbor data.

    def update_self(self, result):
        """
        Updates the data on the current device with the script's result.

        Uses the local `device.my_lock` to ensure thread-safe updates to its own data.

        Args:
            result (object): The result of the script execution.
        """
        with self.device.my_lock: # Acquire local lock for self-updates.
            value = self.device.get_data(self.location) # Get current device data.
            self.device.set_data(self.location, max(result, value)) # Update device data.

    def run(self):
        """
        The main execution loop for the `Worker` thread.

        This loop:
        1. Acquires the location-specific lock from the shared `locations` dictionary.
        2. Collects data from its own device and neighboring devices.
        3. Executes the assigned script with the collected data.
        4. Updates data on its own device and neighboring devices with the script's result.
        5. Releases the location-specific lock.
        """
        # Critical Section: Acquire the lock for the specific location before accessing or modifying data.
        with self.locations[self.location]:
            location_data = [] # List to store collected data.
            self.collect_data(location_data) # Populate location_data.

            if len(location_data) > 0: # Only run script if data was collected.
                result = self.script.run(location_data) # Execute the script.
                self.update_neighbours(result) # Update neighbors' data.
                self.update_self(result)      # Update own data.
            
        # The lock is automatically released when exiting the 'with' block.


class ReusableBarrier():
    """
    A reusable barrier implementation using `threading.Condition`.

    This barrier allows a specified number of threads to synchronize,
    waiting for all threads to reach the barrier before any can proceed.
    It's designed to be reusable for multiple synchronization points.
    """
    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can proceed.
        """
        self.num_threads = num_threads # Total number of threads expected to reach the barrier.
        self.count_threads = self.num_threads # Counter for threads currently at the barrier.
        self.cond = Condition() # Condition variable for blocking and unblocking threads.
 
    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this barrier.
        """
        self.cond.acquire() # Acquire the lock associated with the condition variable.
        self.count_threads -= 1; # Decrement the counter.
        if self.count_threads == 0: # If this is the last thread to reach the barrier.
            self.cond.notify_all() # Wake up all waiting threads.
            self.count_threads = self.num_threads # Reset counter for next use.
        else:
            self.cond.wait(); # Not the last thread, so wait.
        self.cond.release(); # Release the lock.