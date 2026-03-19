"""
This module implements a multi-threaded device simulation system.

It features devices that interact with a supervisor, exchange data with neighbors,
and process assigned scripts using a queue-based worker pattern. The system
uses a shared barrier for global synchronization across devices and per-location
locks for data consistency.

The architecture includes:
- `Device`: Represents a single simulated entity with its own data and threads.
- `DeviceThread`: The main control thread for each `Device`, coordinating supervisor
  interactions, timepoint progression, and managing a pool of parallel workers.
- `ReusableBarrierCond`: (Assumed from import) A barrier synchronization primitive
  that allows a fixed number of threads to wait until all have arrived before proceeding.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond # Assumed to be a custom barrier implementation.

import Queue # Standard Python queue for producer-consumer pattern.

class Device(object):
    """
    Represents a single simulated device in the system.

    Each device manages its own sensor data, interacts with a supervisor,
    and coordinates script execution through its main DeviceThread. Devices
    synchronize globally via a shared barrier and locally using per-location locks.
    """

    def set_shared_barrier(self, shared_barrier):
        """
        Sets the global shared barrier for inter-device synchronization.

        Args:
            shared_barrier (ReusableBarrierCond): The shared barrier instance.
        """
        self.shared_barrier = shared_barrier

    def set_shared_location_locks(self, shared_location_locks):
        """
        Sets the global dictionary of shared locks for data locations.

        Args:
            shared_location_locks (dict): A dictionary mapping locations (int) to Lock objects.
        """
        self.shared_location_locks = shared_location_locks

    def lock_location(self, location):
        """
        Acquires the lock for a specific data location, ensuring exclusive access.
        If the lock for the location does not exist, it's created.

        Note: This method is defined but not explicitly called within the Device class
        or its associated DeviceThread. Its intended use might be external or a remnant
        of a previous design.
        
        Args:
            location (int): The data location to lock.
        """
        if location not in self.shared_location_locks:
            # Lazy initialization of location locks.
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
        self.is_available = Lock()  # Lock to protect this device's sensor_data during read/write.
        self.neighbours = []        # List of neighboring devices.
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal when new scripts have been assigned.
        self.scripts = []           # List of (script, location) tuples assigned to this device.
        self.timepoint_done = Event()   # Event to signal that all scripts for a timepoint have been assigned.
        
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
            shared_barrier = ReusableBarrierCond(len(devices))
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

    def do_in_parallel(self):
        """
        The target function for worker threads. Each worker continuously
        fetches tasks from a shared queue, executes the assigned script
        while ensuring data consistency with locks, and then signals task completion.
        """
        while True:
            # Blocks until a task is available in the queue.
            args = self.queue.get()
            script = args["script"]
            location = args["location"]

            # If a None script is received, it's a shutdown signal.
            if script is None:
                self.queue.task_done() # Mark task as done for queue.join().
                break

            # Acquire the global lock for the specific data location to ensure exclusive access.
            self.device.lock_location(location)

            # Collect data from all neighboring devices for the current location.
            script_data = []
            for device in self.neighbours:
                data = device.get_data(location) # get_data itself uses device.is_available lock.
                if data is not None:
                    script_data.append(data)

            # Collect data from its own device for the current location.
            data = self.device.get_data(location) # get_data itself uses device.is_available lock.

            if data is not None:
                script_data.append(data)

            # If any data was collected, run the script and update devices.
            if script_data != []:
                result = script.run(script_data) # Execute the script.
                
                # Update its own device's data with the script's result.
                # device.is_available lock is acquired/released within set_data.
                self.device.is_available.acquire()
                if location in self.device.sensor_data:
                    self.device.sensor_data[location] = result
                self.device.is_available.release()

                # Update the data in neighboring devices with the script's result.
                # set_data itself uses device.is_available lock.
                for device in self.neighbours:
                    device.set_data(location, result)

            self.device.release_location(location) # Release the global lock for the data location.

            self.queue.task_done() # Mark task as done.

    def __init__(self, device):
        """
        Initializes the DeviceThread and creates its pool of worker threads.

        Args:
            device (Device): The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue.Queue()  # Queue for tasks to be processed by worker threads.
        self.threads = []           # List to hold the worker threads.
        self.neighbours = None      # Stores the list of neighboring devices.

        # Create and start 8 worker threads for parallel script execution.
        for _ in range(8):
            generated_thread = Thread(target=self.do_in_parallel)
            generated_thread.daemon = True # Daemon threads exit when the main program exits.
            self.threads.append(generated_thread)
            generated_thread.start()


    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously processes timepoints: retrieves neighbor information,
        puts assigned scripts into the work queue, waits for their completion,
        and synchronizes globally with other DeviceThreads.
        """
        while True:
            # Retrieves the list of neighboring devices from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()
            
            # If no neighbors are returned (e.g., shutdown signal from supervisor),
            # send shutdown signals to worker threads and exit the loop.
            if self.neighbours is None:
                for _ in range(8): # Put None scripts to signal each worker to stop.
                    self.queue.put({"script":None, "location":None})
                # Wait for all worker threads to finish processing their shutdown signals.
                for thread in self.threads:
                    thread.join()
                self.threads = [] # Clear the list of threads.
                break

            # Waits until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Puts all assigned scripts into the queue for worker threads to process.
            for (script, location) in self.device.scripts:
                self.queue.put({"script":script, "location": location})

            # Blocks until all tasks in the queue (scripts for the current timepoint) are marked as done.
            self.queue.join()

            # Participates in the global barrier synchronization, waiting for all devices
            # to complete their current timepoint processing.
            self.device.shared_barrier.wait()

            self.device.timepoint_done.clear() # Clear the event for the next timepoint.


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