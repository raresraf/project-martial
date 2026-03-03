
"""
This module defines the core components for a distributed device simulation, focusing on
resource management and synchronized script execution across multiple nodes.

It includes:
- Device: Represents an individual node, managing sensor data, scripts, and coordinating with
  a supervisor and other devices. It uses explicit resource locking.
- WorkerThread: Executes assigned scripts at specific data locations, acquiring and releasing
  fine-grained resource locks.
- DeviceThread: The main thread for a Device, orchestrating script execution rounds and
  synchronization across the distributed system.
"""

from threading import Event, Thread, Lock
from barrier import * # Assuming barrier.py contains ReusableBarrierCond and other barrier implementations


class Device(object):
    """
    Represents a single device (node) in a distributed environment.

    This Device manages its own sensor data, processes scripts, and
    coordinates with a supervisor and other devices for synchronized
    operations. It utilizes dynamically created worker threads for
    parallel script execution with explicit resource locking.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance with its unique identity, data, and coordination mechanisms.

        Args:
            device_id (int): A unique integer identifier for this device.
            sensor_data (dict): A dictionary mapping locations (str) to sensor data (object).
            supervisor (object): An object providing central coordination, typically used
                                 to retrieve information about other devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Events for synchronization within the DeviceThread and for script assignment
        self.script_received = Event()      # Signals that a new script or timepoint signal has been assigned.
        self.scripts = []                   # List to store (script, location) tuples assigned to this device.
        self.scriptType = ""                # String indicating if the current signal is a "SCRIPT" or "TIMEPOINT".
        self.timepoint_done = Event()       # Signals that a timepoint has finished processing its scripts.

        # Shared resources managed by a central device (e.g., device_id 0)
        self.allDevices = []                # List of all Device instances in the system.
        self.devices_setup = Event()        # Signals when the initial setup of all devices is complete.

        self.barrierLoop = []               # ReusableBarrierCond instance for timepoint synchronization.
                                            # This is initialized in setup_devices.
        self.canRequestResourcesLock = Lock() # A global lock to serialize access to resource acquisition.
                                              # Used by workers to prevent deadlocks during lock acquisitions.

        # Location-specific locks for this device's data, ensuring mutual exclusion for concurrent access.
        self.myResourceLock = { loc : Lock() for loc in self.sensor_data.keys() }
        
        self.neighbours = []                # List of neighboring devices, updated by DeviceThread from supervisor.
        
        self.numWorkers = 8                 # Configured number of worker threads to be created per timepoint.

        # Create and start the main DeviceThread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a human-readable string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources across all devices, particularly the global barrier.

        This method is typically called by the supervisor during initial system configuration.
        The `ReusableBarrierCond` is initialized based on the total number of devices.

        Args:
            devices (list): A list of all Device instances in the distributed system.
        """
        self.allDevices = devices # Store the reference to all devices.
        # Initialize a conditional reusable barrier, used for synchronizing all devices
        # at the end of each timepoint's processing cycle.
        self.barrierLoop = ReusableBarrierCond(len(devices))
        self.devices_setup.set() # Signal that the initial device setup for this device is complete.


    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device or signals a timepoint completion.

        If a script object is provided, it's added to the device's internal list
        along with its location. If `script` is None, it indicates that no more
        scripts are expected for the current timepoint, and signals `timepoint_done`.

        Args:
            script (object): The script object to be executed, or None to signal timepoint completion.
            location (str): The identifier for the data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the device's list.
            self.scriptType = "SCRIPT"              # Mark the signal type as a script assignment.
            self.script_received.set()              # Signal that a script has been received.
        else:
            self.scriptType = "TIMEPOINT"           # Mark the signal type as timepoint completion.
            self.script_received.set()              # Signal that a timepoint completion signal has been received.
            self.timepoint_done.set()               # Signal that this device is done with its timepoint (for this round of scripts).

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from this device's local storage.

        Args:
            location (str): The identifier of the data location.

        Returns:
            object: The sensor data at the given location, or None if the location is not found.
        """
        # Return data if the location exists in sensor_data, otherwise return None.
        ret = self.sensor_data[location] if location in self.sensor_data else None
        return ret

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location on this device's local storage.

        Args:
            location (str): The identifier of the data location.
            data (object): The new data value to set for the location.
        """
        # Update data only if the location already exists in sensor_data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates a graceful shutdown of the device's main DeviceThread.
        """
        self.thread.join() # Wait for the DeviceThread to complete its execution before shutting down.




class WorkerThread(Thread):
    """
    A worker thread responsible for executing individual scripts assigned to a Device.

    Workers fetch specific scripts from the device's main script list based on provided
    indexes, acquire necessary locks for data integrity (both local and potentially
    neighboring devices), execute the script, and update data.
    """

    def __init__(self, device, listOfIndexes):
        """
        Initializes a WorkerThread.

        Args:
            device (Device): The Device instance this worker belongs to.
            listOfIndexes (list): A list of indices indicating which scripts
                                  from the device's `self.device.scripts` list
                                  this worker is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Name includes device ID for clarity
        self.device = device
        self.listOfIndexes = listOfIndexes # Scripts this worker will process

    def run(self):
        """
        The main execution loop for a WorkerThread.

        For each assigned script index:
        1. Retrieves the script and its location.
        2. Acquires a global resource request lock to prevent deadlocks during
           simultaneous location lock acquisitions from multiple workers across devices.
        3. Acquires location-specific locks for its own data and any relevant neighbor's data.
        4. Releases the global resource request lock.
        5. Gathers relevant data from the current device and its neighbors for the location.
        6. Executes the script with the collected data.
        7. Updates the data on the current device and its neighbors with the script's result.
        8. Releases the previously acquired location-specific locks.
        """
        # Block Logic: Iterate through assigned scripts.
        for i in self.listOfIndexes:
            (script, location) = self.device.scripts[i]

            # Critical Section Entry (Resource Acquisition Phase):
            # Acquire a global lock to serialize the acquisition of location-specific locks
            # across devices, preventing potential deadlocks.
            self.device.allDevices[0].canRequestResourcesLock.acquire()
            
            # Acquire local location lock if applicable.
            if location in self.device.myResourceLock:
                self.device.myResourceLock[location].acquire()
            
            # Acquire location locks from neighboring devices if they manage data at this location.
            for device in self.device.neighbours:
                if self.device.device_id != device.device_id: # Avoid acquiring self-lock again
                      if location in device.myResourceLock:
                            device.myResourceLock[location].acquire()
            self.device.allDevices[0].canRequestResourcesLock.release() # Release global lock after all necessary location locks are acquired.


            script_data = [] # List to accumulate data for script execution.
            
            # Block Logic: Gather data from neighboring devices.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Gather data from the current device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
            
            # Block Logic: Execute script if data is available and update results.
            if script_data != []: # Only run the script if there is data to process.
                # Execute the script with the collected data.
                result = script.run(script_data)
                
                # Update its own sensor data with the script's result.
                self.device.set_data(location, result)
                
                # Propagate the script's result to all relevant neighbors.
                for device in self.device.neighbours:
                    device.set_data(location, result)

            # Critical Section Exit (Resource Release Phase):
            # Release local location lock if applicable.
            if location in self.device.myResourceLock:
                self.device.myResourceLock[location].release()
            
            # Release location locks from neighboring devices if they were acquired.
            for device in self.device.neighbours:
                if self.device.device_id != device.device_id:
                      if location in device.myResourceLock:
                            device.myResourceLock[location].release()



class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for orchestrating timepoint processing.

    It handles global synchronization using a barrier, fetches neighbor information,
    and dynamically creates and manages WorkerThreads for concurrent script execution
    within each timepoint.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.

        This loop continuously:
        1. Waits for the global device setup to be complete.
        2. Enters a main processing loop for timepoints.
        3. Synchronizes with all other devices using a global barrier before starting a new timepoint.
        4. Fetches current neighbor information from the supervisor. If no neighbors (None returned),
           it signifies system shutdown and terminates the thread.
        5. Waits for `script_received` events and processes them. It handles both script assignments
           and timepoint completion signals.
        6. Dynamically creates `WorkerThread`s, distributes the assigned scripts among them,
           starts these workers, and then waits for them to complete. This ensures that
           all scripts for the current timepoint are processed.
        """

        # Wait for the initial setup of all devices to be completed.
        self.device.devices_setup.wait()

        while True: # Main loop for processing timepoints.

            # Global Synchronization Point: Wait for all devices to reach this point
            # before starting the next timepoint processing cycle.
            self.device.allDevices[0].barrierLoop.wait()

            # Block Logic: Retrieve current neighbours and handle shutdown.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # If no neighbours are returned (e.g., supervisor signals termination), break the loop.
            if self.device.neighbours is None:
                break # Exit the timepoint processing loop.
            
            # Block Logic: Process incoming script or timepoint signals.
            while True:
                self.device.script_received.wait()  # Wait for a script assignment or timepoint signal.
                self.device.script_received.clear() # Clear the event for the next signal.
                
                # If the signal was a script, continue waiting for the timepoint completion signal.
                if self.device.scriptType == "SCRIPT":
                    continue # Keep processing script assignments for the current timepoint.
                
                # If the signal was a timepoint completion, wait for all scripts to be assigned,
                # then clear the timepoint_done event and break to start worker execution.
                self.device.timepoint_done.wait()   # Wait for all scripts for this timepoint to be assigned.
                self.device.timepoint_done.clear()  # Clear the event for the next timepoint.
                break # All scripts for the current timepoint have been assigned; proceed to worker execution.

            
            # Block Logic: Dynamic WorkerThread creation and management for parallel script execution.
            workerThreadList = [] # List to hold WorkerThread instances.
            indexesList = []      # List of lists, where each sublist contains script indices for a worker.
        
            # Initialize lists for distributing script indices among workers.
            for i in range(self.device.numWorkers):
                indexesList.append([])
            
            # Distribute script indices cyclically among the worker threads.
            for i in range(len(self.device.scripts)):
                indexesList[i%self.device.numWorkers].append(i) # Assign script 'i' to worker 'i % numWorkers'.
                
            # Create, start, and manage WorkerThreads.
            for i in range(self.device.numWorkers):
                if indexesList[i] != []: # Only create a worker if it has scripts to process.
                    workerThread = WorkerThread(self.device,indexesList[i])
                    workerThreadList.append(workerThread)
                    workerThread.start() # Start the worker thread.

            # Wait for all dynamically created worker threads to complete their tasks for the current timepoint.
            for i in range(self.device.numWorkers):
                if indexesList[i] != []: # Only join workers that were started.
                    workerThreadList[i].join()


