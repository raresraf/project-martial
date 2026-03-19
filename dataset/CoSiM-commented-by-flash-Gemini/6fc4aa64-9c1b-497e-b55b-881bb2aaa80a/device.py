"""
This module implements a multi-threaded device simulation system.

It features devices that interact with a supervisor, exchange data with neighbors,
and process assigned scripts using a custom thread pool mechanism. The system
uses a shared barrier for global synchronization across devices and per-location
locks for data consistency, managed by a designated lead device.

The architecture includes:
- `Device`: Represents a single simulated entity with its own data and threads.
- `DeviceThread`: The main control thread for each `Device`, coordinating supervisor
  interactions, timepoint progression, and managing a pool of `MyThread` workers.
- `MyThread`: Worker threads spawned by `DeviceThread` to execute individual scripts.
- `ReusableBarrierSem`: (Assumed from import `barrier.py`) A barrier synchronization
  primitive, likely implemented using semaphores, that allows a fixed number of threads
  to wait until all have arrived before proceeding.
"""

from threading import Event, Thread, Lock
import supervisor # Assumed to be a module providing supervisor functionality (e.g., get_neighbours).
from barrier import ReusableBarrierSem # Custom barrier class.

import Queue # Standard Python queue, although its direct usage as a Queue.Queue object isn't seen in this file.

class Device(object):
    """
    Represents a single simulated device in the system.

    Each device manages its own sensor data, interacts with a supervisor,
    and coordinates script execution through its main DeviceThread. Devices
    synchronize globally via a shared barrier and locally using per-location locks
    that are initially set up by a lead device (device_id 0).
    """

    def set_shared_barrier(self, shared_barrier):
        """
        Sets the global shared barrier for inter-device synchronization.
        This method is typically called by the lead device (device_id 0) to distribute
        the initialized barrier to all other devices.

        Args:
            shared_barrier (ReusableBarrierSem): The shared barrier instance.
        """
        self.shared_barrier = shared_barrier

    def set_shared_location_locks(self, shared_location_locks):
        """
        Sets the global dictionary (or list) of shared locks for data locations.
        This method is typically called by the lead device (device_id 0) to distribute
        the initialized locks to all other devices.

        Args:
            shared_location_locks (dict/list): A collection mapping locations (int) to Lock objects.
        """
        self.shared_location_locks = shared_location_locks

    def lock_location(self, location):
        """
        Acquires the lock for a specific data location, ensuring exclusive access.
        If the lock for the location does not exist in `self.shared_location_locks`,
        it's created (lazy initialization).

        Note: This method is defined but not explicitly called within the Device class
        or its associated DeviceThread. Its intended use might be external or a remnant
        of a previous design.
        
        Args:
            location (int): The data location to lock.
        """
        if location not in self.shared_location_locks:
            # Lazy initialization of location locks if not already present.
            self.shared_location_locks[location] = Lock()
        self.shared_location_locks[location].acquire()

    def release_location(self, location):
        """
        Releases the lock for a specific data location.

        Note: This method is defined but not explicitly called within the Device class
        or its associated DeviceThread. Its intended use might be external or a remnant
        of a previous design.

        Args:
            location (int): The data location to unlock.
        """
        self.shared_location_locks[location].release()

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping locations (int) to sensor data values.
            supervisor (object): A reference to the supervisor object managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        # `is_available` Lock is declared but not used in `get_data` or `set_data` methods
        # within this class, or by `MyThread`. It appears to be an unused variable.
        self.is_available = Lock()
        self.neighbours = []        # List of neighboring devices (populated by DeviceThread).
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal when new scripts have been assigned.
        self.scripts = []           # List of (script, location) tuples assigned to this device.
        # `timepoint_done` Event is declared but not set or waited upon directly in this module.
        # Its clear() method is called in DeviceThread.run(), but without a preceding set(),
        # this might indicate an unused or incomplete synchronization mechanism.
        self.timepoint_done = Event()   
        
        self.locations = []         # This will hold the shared list of location locks, set by device 0.
        self.get_data_lock = Lock() # Lock specifically for protecting access within the `get_data` method.
        self.ready = Event()        # Event to signal that this device's shared resources are set up.
        self.devices = None         # List of all Device objects in the simulation.
        self.barrier = None         # This will hold the shared global barrier, set by device 0.

        # The main thread responsible for this device's control flow and script delegation.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global shared synchronization primitives (barrier and location locks).
        This method is designed to be called once by all devices, but global initialization
        logic is handled specifically by the device with ID 0.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Only the device with ID 0 initializes and distributes the global shared barrier.
        if self.device_id == 0:
            shared_barrier = ReusableBarrierSem(len(devices))
            for device in devices:
                device.set_shared_barrier(shared_barrier)

        # Only the device with ID 0 initializes and distributes the global shared location locks list.
        if self.device_id == 0:
            shared_location_locks = {}
            # Initialize a fixed number of locks. Hardcoded to 150.
            i = 0
            while i < 150:  
                shared_location_locks[i] = Lock() # Store locks in a dictionary keyed by location.
                i = i + 1

            for dev in devices:
                dev.locations = shared_location_locks # Assign the shared dictionary to each device's `locations`.
                dev.ready.set() # Signal that shared resources are ready for this device.
        

    def assign_script(self, script, location):
        """
        Assigns a processing script and its associated data location to this device.
        If `script` is None, it signals that script assignment for the current
        timepoint is complete (by setting `script_received`).

        Args:
            script (object): The script object (must have a `run` method) to be executed,
                             or None to signal timepoint completion.
            location (int): The data location this script pertains to.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # `timepoint_done` is an Event, but it's set here, and cleared in DeviceThread.run().
            # However, no `timepoint_done.wait()` is found. This suggests incomplete or
            # unused synchronization. For now, it marks the end of script assignment.
            self.timepoint_done.set()

        # Signal that a new script has been received (or timepoint done).
        self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location within this device.
        Uses `get_data_lock` to ensure thread-safe access to `sensor_data`.

        Args:
            location (int): The location ID for which to retrieve data.

        Returns:
            Any: The data at the specified location, or None if the location is not found.
        """
        with self.get_data_lock: # Acquire lock before accessing sensor_data.
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location within this device.
        Note: This method does not use `self.get_data_lock` or `self.is_available`
        to protect `sensor_data` access. This could lead to race conditions if
        `set_data` is called concurrently with `get_data` or other `set_data` calls
        from within the same device, for the same location.

        Args:
            location (int): The location ID for which to set data.
            data (Any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the graceful shutdown sequence for the device's main thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    It is responsible for interacting with the supervisor to get neighbor
    information, managing timepoint progression, and delegating script
    execution to a pool of parallel worker threads via a `Queue`.
    It also participates in global synchronization using a shared barrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # No Queue.Queue object is used here; instead, tasks are directly assigned to MyThread instances.
        self.threads = []           # List to hold the worker threads (MyThread instances).
        self.neighbours = None      # Stores the list of neighboring devices.

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously processes timepoints: retrieves neighbor information,
        spawns and manages worker threads for script execution, and synchronizes
        globally with other DeviceThreads.
        """
        # Wait until the device's shared resources (barrier, location locks) are set up.
        self.device.ready.wait()

        while True:
            # Retrieves the list of neighboring devices from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()
            
            # If no neighbors are returned (e.g., shutdown signal from supervisor),
            # break the loop and terminate this thread.
            if self.neighbours is None:
                # No explicit shutdown for worker threads here since they are created per timepoint.
                break

            # Waits until new scripts have been assigned to this device.
            self.device.script_received.wait()
            self.device.script_received.clear() # Clear the event for the next script assignment.

            rem_scripts = len(self.device.scripts) # Number of scripts to process in this timepoint.

            threads = [] # List to hold MyThread instances for the current timepoint.
            i = 0
            while i < rem_scripts:
                # Create a new MyThread for each script.
                threads.append(MyThread(self.device, self.neighbours, self.device.scripts, i))
                i = i + 1

            # Strategy for executing scripts in parallel.
            # If fewer than 8 scripts, all are started and joined directly.
            if rem_scripts < 8:
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            else:
                # If 8 or more scripts, they are processed in batches of 8.
                pos = 0 # Starting index for the current batch.
                while rem_scripts != 0:
                    if rem_scripts > 8:
                        # Start and join a batch of 8 threads.
                        for i in range(pos, pos + 8):
                            threads[i].start()
                        for i in range(pos, pos + 8):
                            threads[i].join()
                        pos = pos + 8
                        rem_scripts = rem_scripts - 8
                    else:
                        # Start and join the remaining scripts (less than 8).
                        for i in range(pos, pos + rem_scripts):
                            threads[i].start()
                        for i in range(pos, pos + rem_scripts):
                            threads[i].join()
                        rem_scripts = 0 # All scripts processed.

            # Participates in the global barrier synchronization, waiting for all devices
            # to complete their current timepoint processing of scripts.
            self.device.barrier.wait()

            # Clears the `timepoint_done` event. As this event is not `set()` elsewhere
            # in the observable code, this `clear()` operation might be a remnant
            # of a different design or an incomplete synchronization flow.
            self.device.timepoint_done.clear()


class MyThread(Thread):
    """
    A worker thread responsible for executing a single assigned script task.
    These threads are created by DeviceThread for each timepoint's scripts.
    """
    
    def __init__(self, device, neigh, scripts, index):
        """
        Initializes a MyThread instance.

        Args:
            device (Device): The Device instance this worker operates for.
            neigh (list): The list of neighboring devices for the current timepoint.
            scripts (list): The full list of (script, location) tuples for the device.
            index (int): The index in `scripts` that this worker should process.
        """
        Thread.__init__(self, name="Worker for Device %d, Script %d" % (device.device_id, index))
        self.device = device
        self.neigh = neigh          # Neighbors for data exchange.
        self.scripts = scripts      # Reference to the device's full script list.
        self.index = index          # Index of the specific script this thread will run.

    def run(self):
        """
        The main execution logic for `MyThread`.

        It extracts its assigned script, acquires the necessary location lock,
        collects data from neighbors and the local device, executes the script,
        updates data, and then releases the location lock.
        """
        # Extract the specific script and location this thread is responsible for.
        (script, loc) = self.scripts[self.index]
        
        # Acquire the global lock for the specific data location to ensure exclusive access.
        # This prevents race conditions when multiple threads/devices access the same location.
        self.device.locations[loc].acquire()
        
        info = [] # List to collect all relevant data for the script.
        
        # Gathers data from all neighboring devices for the current location.
        for neigh_iter in self.neigh:
            aux_data = neigh_iter.get_data(loc) # `get_data` uses `get_data_lock`.
            if aux_data is not None:
                info.append(aux_data)
        
        # Gathers data from its own device for the current location.
        aux_data = self.device.get_data(loc) # `get_data` uses `get_data_lock`.
        if aux_data is not None:
            info.append(aux_data)
        
        # If any data was collected, run the script and update devices.
        if info != []:
            result = script.run(info) # Execute the script.
            
            # Updates the data in neighboring devices with the script's result.
            # `set_data` here does not use `get_data_lock`, which is an inconsistency.
            for neigh_iter in self.neigh:
                neigh_iter.set_data(loc, result)
            
            # Updates its own device's data with the script's result.
            # `set_data` here does not use `get_data_lock`, which is an inconsistency.
            self.device.set_data(loc, result)
        
        # Releases the global lock for the data location.
        self.device.locations[loc].release()