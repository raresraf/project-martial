


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `ReusableBarrier`: A reusable barrier synchronization mechanism using `threading.Condition`.
- `Device`: Represents a single device, managing sensor data and orchestrating operations.
- `DeviceThread`: The main thread for a `Device`, acting as a producer of scripts for `MyWorker` threads,
  managing neighbor information, and handling timepoint synchronization.
- `MyWorker`: A worker thread that fetches scripts from a `Device`'s queue, executes them,
  and updates device data.

The system utilizes Python's `Queue` for inter-thread communication, `threading.Lock`
for protecting shared data, `threading.Event` for signaling, and a custom
`ReusableBarrier` for coordinating thread execution.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue

class ReusableBarrier(object):
    """
    A reusable barrier synchronization mechanism for multiple threads using `threading.Condition`.
    Threads wait at this barrier until a specified number of threads (`num_threads`) have arrived.
    Once all threads arrive, they are all notified and released simultaneously.
    The barrier can then be reused for subsequent synchronization points.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate
                               in the barrier synchronization.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Counter for threads currently waiting at the barrier.
        self.cond = Condition() # The condition variable used for thread synchronization.

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have arrived. Once all threads are present, they are all released.
        """
        self.cond.acquire() # Acquire the condition variable's intrinsic lock.
        self.count_threads -= 1 # Decrement the count of threads waiting.
        if self.count_threads == 0: # If this is the last thread to arrive:
            self.cond.notify_all() # Notify all waiting threads to resume.
            self.count_threads = self.num_threads # Reset the counter for the next use of the barrier.
        else:
            self.cond.wait() # If not the last thread, wait to be notified.
        self.cond.release() # Release the condition variable's intrinsic lock.

class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution. It uses a `Queue`
    to distribute scripts to `MyWorker` threads and participates in
    global barrier synchronization.
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
        self.script_received = Event() # Event to signal when new scripts are assigned.
        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).
        self.timepoint_done = Event() # Event to signal that the current timepoint's processing is complete.
        self.queue = Queue() # A queue to distribute scripts to MyWorker instances.
        self.setup = Event() # Event to signal that the device's setup is complete (used by DeviceThread).
        self.threads = [] # List to hold the MyWorker instances.
        self.locations_lock = [] # List of shared locks for specific data locations.
        self.barrier = None # Placeholder for the global ReusableBarrier, set in setup_devices.
        self.thread = DeviceThread(self) # The main orchestrating thread for this device.
        self.thread.start() # Start the DeviceThread.

    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global barrier and shared location-specific locks across all devices.
        This method is designed to be called only by the device with `device_id == 0`.
        It creates a single `ReusableBarrier` and a list of `Lock`s for managing access
        to locations, then distributes these and signals setup completion to all devices.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        
        # Block Logic: Only the device with device_id 0 performs this setup.
        if self.device_id == 0:
            # Inline: Creates a global ReusableBarrier for synchronization among all DeviceThreads.
            barrier = ReusableBarrier(len(devices))
            # Inline: Creates 25 shared Lock objects for managing access to locations.
            # Assuming a fixed maximum number of locations (25) as implied by the code.
            for _ in range(25):
                lock = Lock()
                self.locations_lock.append(lock)

            # Block Logic: Distributes the created global barrier and location locks to all devices.
            for device in devices:
                device.barrier = barrier
                device.locations_lock = self.locations_lock # All devices share the same list of locks.
                device.setup.set() # Signal that this device's setup is complete.

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals timepoint completion.
        If a script is provided, it's appended to the device's internal script list.
        If `script` is None, it signals that script assignments for the current
        timepoint are complete.

        Args:
            script (object): The script object to be executed, or None to signal timepoint completion.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the internal list.
            self.script_received.set() # Signal that a new script has been received (likely for DeviceThread).
        else:
            self.timepoint_done.set() # If script is None, signal that script assignments for the timepoint are done.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's `sensor_data` dictionary.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

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
        This implicitly triggers the shutdown of associated `MyWorker` threads managed by `DeviceThread`.
        """
        self.thread.join() # Wait for the DeviceThread to finish its execution.

class DeviceThread(Thread):
    """
    The main orchestrating thread for a `Device`.
    This thread acts as a producer of scripts for `MyWorker` threads,
    fetching neighbor information, and handling timepoint synchronization.
    It manages the lifecycle of its associated `MyWorker` threads.
    """

    def __init__(self, device):
        """
        Initializes a `DeviceThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        It manages the lifecycle of `MyWorker` threads, orchestrates script distribution,
        fetches neighbor information, and handles timepoint synchronization.
        """
        # Block Logic: Wait until the parent `Device` has completed its setup process.
        self.device.setup.wait()

        # Block Logic: Spawn 8 `MyWorker` threads and add them to the device's thread list.
        for _ in range(8):
            thread = MyWorker(self.device)
            thread.start() # Start each worker thread.
            self.device.threads.append(thread) # Add to the list of managed threads.

        while True:
            # Block Logic: Fetch neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Inline: If `neighbours` is None, it signals termination for the device.
            if neighbours is None:
                # Block Logic: Signal all worker threads to terminate by putting `None` into the queue.
                for thread in self.device.threads:
                    for _ in range(8): # Each worker might be waiting on multiple items.
                        self.device.queue.put(None)
                    thread.join() # Wait for each worker thread to finish.
                break # Exit the main loop, terminating the DeviceThread.
            
            # Block Logic: Wait for `timepoint_done` event to be set, indicating scripts are assigned for the current timepoint.
            self.device.timepoint_done.wait()
            # Block Logic: Synchronize with other devices at the global barrier.
            self.device.barrier.wait()
            
            # Block Logic: Distribute assigned scripts to `MyWorker` threads via the shared queue.
            for (script, location) in self.device.scripts:
                # Each work item includes neighbors, location, and the script itself.
                self.device.queue.put((neighbours, location, script))

            self.device.timepoint_done.clear() # Clear the event for the next timepoint.
            
            # Block Logic: Synchronize again at the global barrier, ensuring all devices have finished distributing work.
            self.device.barrier.wait()

class MyWorker(Thread):
    """
    A worker thread that consumes tasks from a `Device`'s shared queue.
    Each task consists of neighbor information, a data location, and a script.
    The worker collects data, executes the script, and updates sensor data
    on relevant devices, ensuring data consistency using location-specific locks.
    """
    
    def __init__(self, device):
        """
        Initializes a `MyWorker` instance.

        Args:
            device (Device): The parent `Device` object this worker belongs to.
        """
        
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        The main execution method for the `MyWorker` thread.
        It continuously fetches tasks (neighbor list, location, script) from the device's queue.
        If a `None` item is received, it signals the thread to terminate.
        Otherwise, it collects data from relevant devices, executes the script,
        and updates their sensor data, all while ensuring thread safety through locks.
        """
        while True:
            # Block Logic: Retrieve a task item from the shared queue.
            # Each item is expected to be a tuple: (neighbours, location, script).
            elem = self.device.queue.get()
            
            # Inline: If `elem` is None, it's a sentinel value indicating thread termination.
            if elem is None:
                break # Exit the loop, terminating the worker thread.
            
            # Block Logic: Acquire the location-specific lock to ensure exclusive access
            # to data at this `location` across all devices during script execution and data update.
            self.device.locations_lock[elem[1]].acquire()
            script_data = [] # List to collect input data for the script.
            data = None # Temporary variable for data retrieval.

            # Block Logic: Collect data from all neighboring devices (provided in `elem[0]`)
            # at the specified location (`elem[1]`).
            for device in elem[0]: # `elem[0]` is the list of neighbors.
                data = device.get_data(elem[1]) # Get data from the neighbor at the specified location.
            if data is not None:
                script_data.append(data) # Add to script input if available.
            
            # Block Logic: Collect data from this worker's own parent device at the specified location.
            data = self.device.get_data(elem[1])
            if data is not None:
                script_data.append(data) # Add to script input if available.

            # Block Logic: If input data is available, execute the script and update device data.
            if script_data != []:
                # Inline: Execute the script's `run` method (`elem[2]`) with the collected data.
                result = elem[2].run(script_data)

                # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.
                for device in elem[0]:
                    device.set_data(elem[1], result) # Update neighbor's data.
                
                self.device.set_data(elem[1], result) # Update this device's own data.
            self.device.locations_lock[elem[1]].release() # Release the location-specific lock.

            self.device.queue.task_done() # Inline: Signal that the current task from the queue has been completed.
