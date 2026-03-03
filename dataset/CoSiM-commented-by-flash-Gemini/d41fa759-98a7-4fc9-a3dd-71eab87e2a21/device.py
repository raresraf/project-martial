"""
This module defines the core components for a distributed device simulation.

It implements a system where each `Device` acts as a node, managing sensor data,
executing scripts, and coordinating with a supervisor and other devices.
The architecture leverages dynamically created `WorkerThread`s for concurrent
script execution, and employs `Lock`s, `Semaphore`s, and a custom `ReusableBarrierSem`
to ensure thread safety and synchronized operations across the distributed system.
"""

from threading import Thread, Lock, Semaphore, Event


class ReusableBarrierSem(object):
    """
    A reusable barrier implementation using Semaphores and a Lock.

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
    `WorkerThread`s for script execution.
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
        
        # Events for internal synchronization and signaling.
        self.script_received = Event()  # Signals when a script has been assigned.
        self.scripts = []               # List to store (script, location) tuples assigned to this device.
        self.timepoint_done = Event()   # Signals that scripts for a timepoint are fully assigned.
        
        # The DeviceThread manages the core logic of this device.
        self.thread = DeviceThread(self)
        self.thread.start()
        
        # Shared synchronization primitives, initialized by `setup_devices`.
        self.barrier = None                 # ReusableBarrierSem for global synchronization across all devices.
        self.lock_neigh = None              # A shared Lock for coordinating updates to neighbor data.
        self.lock_mine = Lock()             # A local Lock for coordinating updates to this device's own data.

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
        the `ReusableBarrierSem` and a shared `lock_neigh` (for neighbor updates),
        then distributes references to these shared objects to all other devices.

        Args:
            devices (list): A list of all Device instances in the distributed system.
        """
        no_devices = len(devices) # Total number of devices in the system.
        lock_neigh = Lock()       # Create a single shared lock for all neighbor updates.
        barrier = ReusableBarrierSem(no_devices) # Create a barrier for all devices.

        # Block Logic: Device 0 (coordinator) distributes the shared barrier and lock.
        if self.device_id == 0:
            for i in range(no_devices):
                devices[i].barrier = barrier    # Assign the shared barrier.
                devices[i].lock_neigh = lock_neigh # Assign the shared neighbor lock.

    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device or signals a timepoint completion.

        If a script object is provided, it's added to the device's internal `scripts` list.
        If `script` is `None`, it signals that no more scripts are expected for the current
        timepoint, setting both `script_received` and `timepoint_done` events.

        Args:
            script (object): The script object to be executed, or `None` to signal timepoint completion.
            location (str): The identifier for the data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
        else:
            # If script is None, it signifies the end of script assignment for this timepoint.
            self.script_received.set() # Signal that script assignment (or termination signal) has been received.
            self.timepoint_done.set()  # Signal that this timepoint is done receiving scripts.

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from this device's local storage.

        Args:
            location (str): The identifier of the data location.

        Returns:
            object: The sensor data at the given location, or `None` if the location is not found.
        """
        data = None
        if location in self.sensor_data:
            data = self.sensor_data[location] # Retrieve data if location exists.
        return data


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


class WorkerThread(Thread):
    """
    A worker thread dynamically created to execute a single script for a `Device`.

    This thread is responsible for collecting data from its device and neighbors,
    executing a specific script, and then updating the data on its device and neighbors.
    It uses shared locks for managing concurrent data updates.
    """

    def __init__(self, device, script, location, neighbours):
        """
        Initializes a `WorkerThread`.

        Args:
            device (Device): The `Device` instance this worker belongs to.
            script (object): The script object to be executed.
            location (str): The identifier for the data location the script operates on.
            neighbours (list): A list of neighboring `Device` instances.
        """
        Thread.__init__(self, name="Worker Thread")
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def collect_data(self, location_data):
        """
        Collects relevant data for the script from the current device and its neighbors.

        Args:
            location_data (list): A list to append the collected data to.
        """
        location_data.append(self.device.get_data(self.location)) # Get data from self.
        for i in range(len(self.neighbours)):
            data = self.neighbours[i].get_data(self.location) # Get data from neighbor.
            location_data.append(data) # Append to the list.

    def update_neighbours(self, result):
        """
        Updates the data on neighboring devices with the script's result.

        Uses the shared `device.lock_neigh` to ensure thread-safe updates to neighbor data.

        Args:
            result (object): The result of the script execution.
        """
        no_neigh = len(self.neighbours)
        for i in range(no_neigh):
            with self.device.lock_neigh: # Acquire shared lock for neighbor updates.
                value = self.neighbours[i].get_data(self.location) # Get current neighbor data.
                self.neighbours[i].set_data(self.location, max(result, value)) # Update neighbor data.

    def update_self(self, result):
        """
        Updates the data on the current device with the script's result.

        Uses the local `device.lock_mine` to ensure thread-safe updates to its own data.

        Args:
            result (object): The result of the script execution.
        """
        with self.device.lock_mine: # Acquire local lock for self-updates.
            value = self.device.get_data(self.location) # Get current device data.
            self.device.set_data(self.location, max(result, value)) # Update device data.

    def run(self):
        """
        The main execution loop for the `WorkerThread`.

        This loop:
        1. Collects data from the device and its neighbors.
        2. Executes the assigned script with the collected data.
        3. Updates data on its own device and neighboring devices with the script's result.
        """
        location_data = [] # List to store collected data.
        self.collect_data(location_data) # Populate location_data.

        if len(location_data) > 0: # Only run script if data was collected.
            result = self.script.run(location_data) # Execute the script.
            self.update_neighbours(result) # Update neighbors' data.
            self.update_self(result)      # Update own data.


class DeviceThread(Thread):
    """
    The main thread for a `Device`, responsible for orchestrating timepoint processing.

    It manages synchronization using a global `ReusableBarrierSem`, fetches neighbor
    information from the supervisor, and dynamically creates and manages `WorkerThread`s
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
        2. Waits for scripts to be assigned via `script_received` event.
        3. Dynamically creates `WorkerThread`s for each assigned script, starts them,
           and then waits for their completion.
        4. Synchronizes with all other devices using the global `ReusableBarrierSem`.
        5. Waits for the `timepoint_done` event (signaling all scripts are assigned
           and processed), then clears it for the next timepoint.
        """
        threads = [None] * 200 # Pre-allocates a list for worker threads (max 200 scripts per timepoint).
        while True:
            # Block Logic: Retrieve current neighbours and handle system shutdown.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Check for a shutdown signal from the supervisor.
                break # Exit the timepoint processing loop.

            # Synchronization Point: Wait for new scripts to be assigned for this timepoint.
            self.device.script_received.wait()  # Blocks until `assign_script` sets this event.
            self.device.script_received.clear() # Clear the event for the next timepoint.
            
            # Block Logic: Create and manage WorkerThreads for each assigned script.
            for i in range(len(self.device.scripts)):
                (script, location) = self.device.scripts[i]
                # Create a new WorkerThread for each script.
                threads[i] = WorkerThread(self.device, script, location, neighbours)
                threads[i].start() # Start the worker thread.

            # Wait for all dynamically created worker threads to complete their tasks for the current timepoint.
            for i in range(len(self.device.scripts)):
                threads[i].join() # Block until the worker thread finishes.

            # Global Synchronization Point: Wait for all devices to reach this barrier
            # before advancing to the next timepoint.
            self.device.barrier.wait()
            
            # Synchronization Point: Wait for all scripts to be fully processed for this timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Clear the event for the next timepoint.