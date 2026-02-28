

"""
This module provides components for simulating a device within a distributed system.
It features a multi-threaded architecture where each `Device` instance operates
within its `DeviceThread`. This `DeviceThread` in turn dispatches tasks (scripts)
to `Worker` threads for concurrent execution. Synchronization between devices
is managed through a custom `ReusableBarrier` class, utilizing Python's `Condition`
primitive, and `Lock` objects for location-specific data access.

Key Components:
- `Device`: Represents an individual simulated device, managing sensor data and scripts.
- `DeviceThread`: The main thread for a device, orchestrating script execution via worker threads.
- `Worker`: Executes individual scripts, handling data acquisition and propagation with locking.
- `ReusableBarrier`: A synchronization primitive allowing multiple threads to wait for each other.
"""

from threading import Event, Thread, Condition, Lock


class Device(object):
    """
    Represents an individual simulated device in a distributed system.
    Each `Device` instance manages its own sensor data, communicates with a
    central supervisor, and processes assigned scripts. It utilizes a dedicated
    `DeviceThread` to manage its operations and participates in synchronization
    with other devices through shared locks and a `ReusableBarrier`.
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
        self.scripts = []            # List to store (script, location) tuples.
        self.scripts_done = Event()  # Event to signal when all scripts are assigned for a timepoint.
        self.my_lock = Lock()        # A lock for protecting this device's internal data during access by workers.

        # These will be initialized during setup_devices.
        self.locations = None        # Dictionary of Locks for location-specific data access.
        self.barrier = None          # Shared barrier for synchronizing device threads.

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives (location-specific locks and a barrier)
        across a group of devices. This method ensures that these shared resources
        are created only once by the device with `device_id == 0` and then
        distributed among all participating devices.

        Args:
            devices (list): A list of Device objects that are part of the same group.
        """
        # Block Logic: Device with ID 0 is responsible for initializing shared resources.
        if self.device_id is 0:
            self.locations = {}                               # Initialize dictionary for location locks.
            self.barrier = ReusableBarrier(len(devices));     # Create a new barrier for the group.
            # Block Logic: Create a Lock for each unique location in this device's sensor data.
            # This ensures only one worker thread can access/modify a specific location's data at a time.
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass # Functional Utility: Skip if lock already exists for this location.
                else:
                    self.locations[loc] = Lock()
        # Block Logic: For all other devices (device_id != 0), acquire references to the shared resources
        # initialized by device 0.
        else:
            self.locations = devices[0].locations      # Reference the shared location locks from device 0.
            self.barrier = devices[0].get_barrier()    # Reference the shared barrier from device 0.
            # Block Logic: Add any unique locations from this device's sensor data to the shared locks if not present.
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass
                else:
                    self.locations[loc] = Lock()

        # Functional Utility: Create and start the dedicated thread for this device.
        self.thread = DeviceThread(self, self.barrier, self.locations)
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location on this device.
        If `script` is `None`, it signals that all scripts for the current timepoint are assigned.

        Args:
            script (object or None): The script object to execute, or `None`.
            location (str): The identifier for the location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_done.set() # Signal that all scripts for this timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's internal state.

        Args:
            location (str): The identifier of the location for which to retrieve data.

        Returns:
            Any: The sensor data associated with the location, or `None` if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location on this device.

        Args:
            location (str): The identifier of the location to update.
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

    def get_barrier(self):
        """
        Returns the shared `ReusableBarrier` instance used by this group of devices.

        Returns:
            ReusableBarrier: The shared barrier instance.
        """
        return self.barrier

class DeviceThread(Thread):
    """
    The dedicated operational thread for a `Device` instance. It orchestrates
    the device's simulation lifecycle, including fetching neighbor information
    from the supervisor, waiting for scripts to be assigned, dispatching these
    scripts to `Worker` threads for concurrent execution, and synchronizing
    with other device threads using a shared `ReusableBarrier`.
    """

    def __init__(self, device, barrier, locations):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread is associated with.
            barrier (ReusableBarrier): The shared barrier for synchronizing with other devices.
            locations (dict): A dictionary of `Lock` objects for location-specific data access.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)

        self.device = device
        self.barrier = barrier
        self.locations = locations

    def run(self):
        """
        The main execution loop for the device's thread.
        It continuously processes timepoints, managing script execution and synchronization.
        """
        # Block Logic: Main loop for continuous simulation timepoints.
        while True:
            # Pre-condition: Fetch information about neighboring devices from the supervisor.
            # This also serves as a signal for the simulation's continuation or termination.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit the loop if the supervisor signals simulation end.

            # Block Logic: Wait for all scripts for the current timepoint to be assigned to this device.
            self.device.scripts_done.wait()
            self.device.scripts_done.clear() # Reset the event for the next timepoint.
            
            workers = [] # List to hold worker threads for current timepoint's scripts.
            # Block Logic: Create and start a worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                w = Worker(self.device, neighbours, script, location, self.locations)
                workers.append(w)
                w.start()

            # Block Logic: Wait for all worker threads to complete their assigned scripts for this timepoint.
            for w in workers:
                w.join()

            # Functional Utility: Synchronize with all other device threads using the shared barrier.
            # All device threads must reach this point before any can proceed to the next timepoint.
            self.barrier.wait()

class Worker(Thread):
    """
    A worker thread spawned by `DeviceThread` to execute a single assigned script.
    It manages access to location-specific data using shared locks, gathers
    necessary sensor data from its own and neighboring devices, runs the script,
    and then propagates the computed results back to the relevant devices.
    """
    def __init__(self, device, neighbours, script, location, locations):
        """
        Initializes a Worker thread.

        Args:
            device (Device): The Device instance this worker thread is associated with.
            neighbours (list): A list of neighboring Device objects to gather data from.
            script (object): The script object to be executed.
            location (str): The specific location identifier the script pertains to.
            locations (dict): A shared dictionary of `Lock` objects for location-specific data access.
        """
        Thread.__init__(self, name="Worker for Device %d, Location %s" % (device.device_id, location))
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.locations = locations

    def run(self):
        """
        Executes the assigned script.
        This involves acquiring location-specific locks, gathering data,
        running the script, updating data on involved devices, and releasing locks.
        """
        # Block Logic: Acquire the lock for the specific location.
        # This ensures exclusive access to the data associated with 'self.location'
        # across all workers from different devices that might operate on the same location.
        self.locations[self.location].acquire()
        script_data = [] # List to accumulate data for the script.
        
        # Block Logic: Gather data from all neighboring devices for the current location.
        # Pre-condition: Each neighbor's internal data is protected by its `my_lock`.
        for device in self.neighbours:
            device.my_lock.acquire() # Acquire lock to safely access neighbor's data.
            data = device.get_data(self.location)
            device.my_lock.release() # Release neighbor's lock.
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Gather data from the current device itself for the current location.
        # Pre-condition: This device's internal data is protected by its `my_lock`.
        self.device.my_lock.acquire() # Acquire lock to safely access this device's data.
        data = self.device.get_data(self.location)
        self.device.my_lock.release() # Release this device's lock.
        if data is not None:
            script_data.append(data)

        # Pre-condition: Check if any data was collected before running the script.
        if script_data != []:
            # Functional Utility: Execute the script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Propagate the result of the script execution back to all neighboring devices.
            # Pre-condition: Each neighbor's internal data is protected by its `my_lock`.
            for device in self.neighbours:
                device.my_lock.acquire() # Acquire lock to safely update neighbor's data.
                device.set_data(self.location, result)
                device.my_lock.release() # Release neighbor's lock.

            # Block Logic: Update the current device's own sensor data with the result.
            # Pre-condition: This device's internal data is protected by its `my_lock`.
            self.device.my_lock.acquire() # Acquire lock to safely update this device's data.
            self.device.set_data(self.location, result)
            self.device.my_lock.release() # Release this device's lock.
            
        # Block Logic: Release the lock for the specific location.
        self.locations[self.location].release()



class ReusableBarrier():
    """
    A reusable barrier synchronization primitive that allows a fixed number of threads
    to wait for each other to reach a common execution point (the barrier).
    Once all threads have arrived, they are all released to proceed.
    It uses a `threading.Condition` object to manage thread blocking and notification,
    and resets itself after each synchronization point for reusability.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier with a specified number of participating threads.

        Args:
            num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads    # Current count of threads yet to reach the barrier.
        self.cond = Condition()                  # Condition variable for blocking and waking threads.
 
    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have also called `wait()`.
        Once all threads have arrived, they are all released to proceed.
        The barrier then resets for subsequent use.
        """
        self.cond.acquire()                      # Acquire the lock associated with the condition variable.
        self.count_threads -= 1;                 # Decrement the count of threads yet to arrive.
        # Block Logic: Check if this is the last thread to arrive at the barrier.
        if self.count_threads == 0:
            self.cond.notify_all()               # If all threads have arrived, notify all waiting threads.
            self.count_threads = self.num_threads    # Reset the counter for the next cycle of the barrier.
        else:
            self.cond.wait();                    # If not the last thread, wait (release lock and block) until notified.
        self.cond.release();                     # Release the lock associated with the condition variable.