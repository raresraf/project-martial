




"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `Device`: Represents a single device, managing its sensor data and orchestrating operations.
- `DeviceThread`: The main thread for a `Device`, responsible for managing work distribution
  and spawning `Worker` threads.
- `Workpool`: A thread-safe queue for distributing scripts/tasks to worker threads.
- `Worker`: A thread that fetches scripts from the `Workpool`, executes them, and updates data.

The system relies on synchronization primitives from the `threading` module and a custom
`barrier` module (specifically `ReusableBarrierCond`) for coordinating thread execution
and data access across devices and within a single device.
"""

from threading import Thread, Lock
import barrier

class Device(object):
    """
    Represents a single device within the simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution through its `DeviceThread`.
    It uses a `Workpool` to manage scripts and `locationslocks` for data consistency.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for the device.
            supervisor (object): A reference to a supervisor object for inter-device communication.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).
        self.barrier = None # Placeholder for the global ReusableBarrierCond, set in setup_devices.
        self.locationslocks = {} # Dictionary of locks, one for each data location, shared across devices.
        self.neighbours = [] # List to store references to neighboring devices.
        self.workpool = Workpool() # The Workpool instance for managing scripts.
        self.thread = DeviceThread(self) # The main thread for this device.

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared location-specific locks and a global barrier
        across all devices in the simulation. This method is designed to be called only
        by the device with `device_id == 0`.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        # Block Logic: Only the device with device_id 0 performs this setup.
        if self.device_id == 0:
            locationslocks = {} # Dictionary to store shared locks for each data location.
            
            # Inline: Initialize a unique `Lock` for each distinct data location present across all devices.
            for dev in devices:
                for location in dev.sensor_data:
                    locationslocks[location] = Lock()

            # Inline: Creates a global ReusableBarrierCond for synchronization among all DeviceThreads.
            barr = barrier.ReusableBarrierCond(len(devices)) # The barrier expects one `DeviceThread` per `Device`.

            # Inline: Distributes the created global `locationslocks` and `ReusableBarrierCond` to all devices
            # and starts their respective `DeviceThread`s.
            for dev in devices:
                dev.locationslocks = locationslocks
                dev.barrier = barr
                dev.thread.start() # Start the main thread for each device.

    def assign_script(self, script, location):
        """
        Assigns a script to the device and adds it to the device's workpool.
        If `script` is None, it signals the end of work for the workpool.

        Args:
            script (object): The script object to be executed.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            # Inline: Append the script and its location to the device's internal list of scripts.
            self.scripts.append((script, location))

            # Inline: Add the script and location as work to the workpool for worker threads to pick up.
            self.workpool.putwork(script, location)
        else:
            # Inline: If script is None, signal to the workpool that no more work will be added.
            self.workpool.endwork()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's `sensor_data` dictionary.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location in this device's `sensor_data` dictionary.
        The data is updated only if the location exists in the `sensor_data`.

        Args:
            location (int): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by waiting for its main `DeviceThread` to complete.
        """
        self.thread.join() # Wait for the DeviceThread to finish its execution.


class DeviceThread(Thread):
    """
    The main thread for a `Device`. It acts as an orchestrator, responsible for
    interacting with the device's `Workpool`, fetching neighbor information,
    and managing a pool of `Worker` threads to execute scripts concurrently.
    It also participates in global barrier synchronization.
    """

    def __init__(self, device):
        """
        Initializes a `DeviceThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = [] # List to hold spawned `Worker` threads.
        
        # A barrier specifically for synchronizing the 8 `Worker` threads and this `DeviceThread`.
        # Assuming 8 workers + 1 DeviceThread = 9 participants.
        self.workerbar = barrier.ReusableBarrierCond(9)

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        It continuously manages the lifecycle of script processing, including
        distributing scripts to a workpool, fetching neighbor information,
        spawning and managing worker threads, and synchronizing at various barriers.
        """
        while True:
            # Block Logic: Put all currently assigned scripts into the workpool for worker threads.
            # This prepares the scripts for concurrent processing.
            self.device.workpool.putlistwork(self.device.scripts)

            # Block Logic: Fetch neighbor information from the supervisor.
            # This information is crucial for workers to collect data from adjacent devices.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            neighbours = self.device.neighbours # Store a local reference for easier access.

            # Inline: If `neighbours` is None, it signals termination.
            if neighbours is None:
                break # Exit the loop, signaling device shutdown.

            # Block Logic: Spawn 8 `Worker` threads to process scripts from the workpool.
            # Each worker will fetch tasks from the shared workpool.
            for i in xrange(8):
                worker = Worker(self.workerbar, self.device) # Create a new Worker instance.
                self.workers.append(worker) # Add the worker to the list.
                self.workers[i].start() # Start the worker thread.

            # Block Logic: Wait for all 8 `Worker` threads and this `DeviceThread` to reach their barrier.
            # This ensures that all workers have started processing or determined there is no more work.
            self.workerbar.wait()

            # Block Logic: Join (wait for) all `Worker` threads to complete their current tasks.
            for i in range(8):
                self.workers[i].join()
            del self.workers[:] # Clear the list of workers for the next cycle.

            # Block Logic: Wait at the global device barrier.
            # This synchronizes all `DeviceThread` instances across all `Device`s before proceeding.
            self.device.barrier.wait()


class Workpool(object):
    """
    A thread-safe workpool for managing and distributing scripts/tasks to worker threads.
    It acts as a shared queue where `DeviceThread` can put work, and `Worker` threads
    can retrieve work to process concurrently.
    """

    def __init__(self):
        """
        Initializes the Workpool.
        """
        self.scripts = [] # List to store (script, location) tuples, representing work items.
        self.lock = Lock() # A lock to ensure thread-safe access to the `scripts` list and `done` flag.
        self.done = False # A flag indicating whether all work has been put into the pool.

    def getwork(self):
        """
        Retrieves a single work item (script and location pair) from the workpool.
        This method is thread-safe.

        Returns:
            tuple or None: A (script, location) tuple if work is available,
                           an empty tuple `()` if the pool is empty but not yet `done`,
                           or `None` if the pool is `done` and empty.
        """
        self.lock.acquire() # Acquire the lock to ensure exclusive access.
        
        # Block Logic: Check if there's still potential for more work or if the current queue has items.
        if self.done is False or len(self.scripts) > 0:
            
            if len(self.scripts) > 0: # If there are scripts in the queue:
                pair = self.scripts.pop() # Retrieve the last script from the queue.
                self.lock.release() # Release the lock.
                return pair
            else: # If the queue is empty but not yet `done`:
                self.lock.release() # Release the lock.
                return () # Return an empty tuple to signal no work currently, but more might come.
        else: # If the pool is `done` and empty:
            self.lock.release() # Release the lock.
            return None # Return None to signal permanent end of work.

    def putwork(self, script, location):
        """
        Adds a single work item (script and location pair) to the workpool.
        This method is thread-safe.

        Args:
            script (object): The script object to add.
            location (int): The location associated with the script.
        """
        self.lock.acquire() # Acquire the lock.
        self.scripts.append((script, location)) # Add the work item.
        self.lock.release() # Release the lock.

    def endwork(self):
        """
        Signals that no more work will be added to the workpool.
        This is used by `Worker` threads to know when to terminate.
        This method is thread-safe.
        """
        self.lock.acquire() # Acquire the lock.
        self.done = True # Set the `done` flag.
        self.lock.release() # Release the lock.

    def putlistwork(self, scripts):
        """
        Replaces the current list of scripts in the workpool with a new list
        and resets the `done` flag. This is used to load a new batch of work.
        This method is thread-safe.

        Args:
            scripts (list): A new list of (script, location) tuples to populate the workpool.
        """
        self.lock.acquire() # Acquire the lock.
        self.done = False # Reset the `done` flag as new work is being added.
        self.scripts = list(scripts) # Replace the scripts with the new list.
        self.lock.release() # Release the lock.


class Worker(Thread):
    """
    A worker thread that fetches tasks (scripts and locations) from a shared `Workpool`,
    executes them, and updates sensor data on relevant devices.
    It participates in a local barrier (`workerbar`) for synchronization with its `DeviceThread` parent.
    """
    def __init__(self, barr, device):
        """
        Initializes a `Worker` thread.

        Args:
            barr (barrier.ReusableBarrierCond): The `ReusableBarrierCond` for synchronizing
                                                with the `DeviceThread` and other `Worker` threads.
            device (Device): The parent `Device` object this worker belongs to.
        """
        
        Thread.__init__(self, name="Worker Thread")
        self.lock = Lock() # This lock appears to be unused in the current implementation of Worker.
        self.barrier = barr
        self.device = device

    def run(self):
        """
        The main execution method for the `Worker` thread.
        It continuously fetches work (scripts) from the `Device`'s `Workpool`,
        processes it, and handles termination signals.
        """
        while True:
            # Block Logic: Retrieve a work item from the shared workpool.
            # `getwork()` can return a (script, location) tuple, an empty tuple `()`, or `None`.
            work = self.device.workpool.getwork()

            if work is None: # Inline: If `getwork()` returns `None`, it signals the permanent end of work.
                # Inline: Wait at the worker-specific barrier before terminating,
                # ensuring synchronization with the `DeviceThread`.
                self.barrier.wait() 
                return # Terminate the worker thread.
            else:
                # Inline: If work is available (not an empty tuple), process it.
                if work is not (): 
                    self.update(work[0], work[1]) # Call the update method to execute the script.

    def update(self, script, location):
        """
        Executes a given script at a specific location, collecting necessary data
        from the current device and its neighbors, and then updates their sensor data
        with the script's result.

        Args:
            script (object): The script object to be executed.
            location (int): The data location identifier to which the script applies.
        """
        # Block Logic: Acquire the shared lock for this specific location to ensure exclusive access
        # during data collection and update operations.
        self.device.locationslocks[location].acquire()

        script_data = [] # List to collect input data for the script.
        
        # Block Logic: Collect data from all neighboring devices at the specified location.
        for device in self.device.neighbours:
            data = device.get_data(location) # Get data from the neighbor.
            if data is not None:
                script_data.append(data) # Add to script input if available.
        
        # Block Logic: Collect data from this worker's own parent device at the specified location.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data) # Add to script input if available.

        # Block Logic: If input data is available, execute the script and update device data.
        if script_data != []:
            # Inline: Execute the script's `run` method with the collected data.
            result = script.run(script_data)

            # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.
            for device in self.device.neighbours:
                device.set_data(location, result) # Update neighbor's data.
            
            self.device.set_data(location, result) # Update this device's own data.

        self.device.locationslocks[location].release() # Release the shared lock for this location.
