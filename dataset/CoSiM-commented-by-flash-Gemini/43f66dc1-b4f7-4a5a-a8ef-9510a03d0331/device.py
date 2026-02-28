
"""
This module provides components for simulating a device within a distributed system.
It employs a multi-threaded architecture where each `Device` instance operates
within its `DeviceThread`. The `DeviceThread` coordinates `ParallelScript` workers
to execute scripts concurrently. Synchronization among devices is managed through
an inline `ReusableBarrier` class, and data access to locations is protected by
a list of `threading.Lock` objects (`big_lock`).

Key Components:
- `ReusableBarrier`: A synchronization primitive allowing multiple threads to wait for each other.
- `Device`: Represents an individual simulated device, managing sensor data and scripts.
- `ParallelScript`: Worker threads that execute individual scripts on specific locations.
- `DeviceThread`: The main thread for a device, orchestrating script execution and synchronization.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A reusable barrier synchronization primitive that enables a fixed number of
    threads to pause their execution until all threads have reached a designated
    synchronization point. It uses two alternating phases to ensure correct
    reusability and prevent premature releases.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that must arrive at the barrier.
        """
        self.num_threads = num_threads
        # Two counters to manage the two phases of the barrier, ensuring reusability.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()          # A lock to protect access to the thread counters.
        # Two semaphores, one for each phase, to block and release threads.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this point.
        It orchestrates the two phases of the barrier to ensure proper synchronization.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the mutable count of threads
                                  remaining in the current phase.
            threads_sem (Semaphore): The semaphore corresponding to the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1  # Decrement the count of threads yet to arrive.
            # Block Logic: If this is the last thread to arrive in the current phase.
            if count_threads[0] == 0:
                # Functional Utility: Release all waiting threads for this phase.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads # Reset the counter for the next cycle.
        # Block Logic: Acquire the semaphore, blocking the thread until it's released by the last arriving thread.
        threads_sem.acquire()


class Device(object):
    """
    Represents an individual simulated device in a distributed system.
    Each `Device` instance manages its own sensor data, communicates with a
    central supervisor, and processes assigned scripts. It utilizes a dedicated
    `DeviceThread` to manage its operations and coordinates with other devices
    through shared synchronization primitives (`ReusableBarrier` and `Lock` objects).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping location IDs to their
                                current sensor data values.
            supervisor (object): A reference to the central supervisor managing
                                 the distributed system.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a script has been received.
        self.scripts = []             # List to store (script, location) tuples.

        self.timepoint_done = Event()  # Event to signal completion of a timepoint's tasks.
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations.
        self.thread.start()
        self.barrier = None           # Shared barrier for synchronizing device threads.
        self.big_lock = []            # List of Locks for location-specific data access.

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives (a `ReusableBarrier` and a
        list of `Lock` objects for locations) across a group of devices. This
        ensures these resources are created once and shared among all devices.

        Args:
            devices (list): A list of Device objects that are part of the same group.
        """
        barrier = ReusableBarrier(len(devices)) # Create a new barrier for the group.
        lock1 = Lock() # A general lock (appears unused in the provided code, but kept for context).

        num_locations = {} # Temporary dictionary to collect all unique locations across devices.

        # Block Logic: Populate num_locations with all unique location keys from all devices.
        for device in devices:
            for location in device.sensor_data.keys():
                num_locations[location] = 1 # Value doesn't matter, just need unique keys.

        # Functional Utility: Create a list of Locks, one for each unique location.
        # This list (`big_lock`) will be shared across all devices to protect location-specific data.
        big_lock = [Lock() for _ in range(len(num_locations))]

        # Block Logic: Distribute the shared barrier and the list of location locks to all devices.
        for device in devices:
            device.lock1 = lock1 # Assign the general lock.
            device.barrier = barrier # Assign the shared barrier.
            device.big_lock = big_lock # Assign the shared list of location locks.

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location on this device.
        If `script` is `None`, it signals that all scripts for the current timepoint are assigned.

        Args:
            script (object or None): The script object to execute, or `None`.
            location (int): The integer identifier for the location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a script has been received.
        else:
            self.timepoint_done.set() # Signal that all scripts for this timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's internal state.

        Args:
            location (int): The integer identifier of the location for which to retrieve data.

        Returns:
            Any: The sensor data associated with the location, or `None` if not found.
        """
        return self.sensor_data[location] \
            if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location on this device.

        Args:
            location (int): The integer identifier of the location to update.
            data (Any): The new sensor data value for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device's operational thread.
        This method waits for the device's thread to complete its execution.
        """
        self.thread.join()


class ParallelScript(Thread):
    """
    A worker thread responsible for executing a specific script for a designated
    location. It acquires necessary locks to ensure safe data access, gathers
    sensor data from its associated device and its neighbors, executes the script,
    and then updates the relevant data on all involved devices.
    """
    
    def __init__(self, device, scripts, location, neighbours):
        """
        Initializes a ParallelScript worker.

        Args:
            device (Device): The Device instance this worker thread is associated with.
            scripts (list): A list of script objects to be executed sequentially for this location.
            location (int): The integer identifier of the location the script pertains to.
            neighbours (list): A list of neighboring Device objects to gather data from.
        """
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the assigned scripts for its location.
        This involves acquiring location-specific locks, gathering data,
        running scripts, updating device data, and releasing locks.
        """
        # Block Logic: Iterate through each script assigned to this worker for the current location.
        for script in self.scripts:
            # Pre-condition: Acquire the lock for the specific location.
            # This ensures exclusive access to the data associated with 'self.location'
            # across all threads (including workers from other devices) operating on the same location.
            self.device.big_lock[self.location].acquire()

            script_data = [] # List to accumulate data for the script.
            
            # Block Logic: Gather data from all neighboring devices for the current location.
            # No individual locks are acquired here for neighbors as this `big_lock`
            # at the location level should be sufficient if `get_data` and `set_data`
            # internally handle their own consistency. However, in this code, `my_lock`
            # is used internally within `Worker` when interacting with `device.get_data` / `device.set_data`.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Gather data from the current device itself for the current location.
            data = self.device.get_data(self.location)

            if data is not None:
                script_data.append(data)

            # Pre-condition: Check if any data was collected before running the script.
            if script_data != []:
                # Functional Utility: Execute the script with the collected data.
                result = script.run(script_data)

                # Block Logic: Propagate the result of the script execution back to all neighboring devices.
                for device in self.neighbours:
                    device.set_data(self.location, result)

                # Block Logic: Update the current device's own sensor data with the result.
                self.device.set_data(self.location, result)
            
            # Block Logic: Release the lock for the specific location.
            self.device.big_lock[self.location].release()


class DeviceThread(Thread):
    """
    The dedicated operational thread for a `Device` instance. It orchestrates
    the device's simulation lifecycle, including fetching neighbor information
    from the supervisor, waiting for scripts to be assigned, organizing these
    scripts by location, dispatching them to `ParallelScript` worker threads
    for concurrent execution, and synchronizing with other device threads
    using a shared `ReusableBarrier`.
    """
    

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device's thread.
        It continuously processes timepoints, managing script execution and synchronization.
        """
        while True:
            # Pre-condition: Fetch information about neighboring devices from the supervisor.
            # This also serves as a signal for the simulation's continuation or termination.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit the loop if the supervisor signals simulation end.

            # Block Logic: Wait for all scripts for the current timepoint to be assigned to this device.
            self.device.timepoint_done.wait()

            threads = [] # List to hold ParallelScript worker threads.
            scripts = {} # Dictionary to group scripts by location.

            # Block Logic: Organize assigned scripts by their location.
            # Invariant: Each (script, location) tuple is processed.
            for (script, location) in self.device.scripts:
                if scripts.has_key(location): # Functional Utility: Python 2 dict method.
                    scripts[location].append(script)
                else:
                    scripts[location] = [script]

            # Block Logic: Create and start a ParallelScript worker thread for each unique location.
            # Each worker will handle all scripts for its specific location.
            for location in scripts.keys():
                new = ParallelScript(self.device, scripts[location],
                                     location, neighbours)
                threads.append(new)

            # Block Logic: Start all ParallelScript worker threads.
            for thread in threads:
                thread.start()

            # Block Logic: Wait for all ParallelScript worker threads to complete.
            for thread in threads:
                thread.join()

            # Functional Utility: Synchronize with all other device threads using the shared barrier.
            # All device threads must reach this point before any can proceed to the next timepoint.
            self.device.barrier.wait()

            # Functional Utility: Clear the timepoint_done event, resetting it for the next timepoint.
            self.device.timepoint_done.clear()
