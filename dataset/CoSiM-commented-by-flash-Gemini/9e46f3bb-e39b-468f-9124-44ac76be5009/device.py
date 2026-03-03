
"""
This module defines the core components for a distributed device simulation or data processing system.

It includes:
- Device: Represents an individual node in the distributed system, managing sensor data, scripts, and coordinating with other devices.
- DeviceThread: The main thread for a Device, responsible for orchestrating script execution rounds and synchronization.
- Worker: A thread that executes assigned scripts on specific data locations.
"""

from threading import Event, Thread, Lock
from Queue import Queue
from barrier import ReusableBarrier


class Device(object):
    """
    Represents a single device (node) in a distributed environment.

    Each device manages its own sensor data, processes scripts, and
    coordinates with a supervisor and other devices for synchronized
    operations. It utilizes multiple worker threads for parallel script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary containing sensor data,
                                 where keys are locations and values are data.
            supervisor (object): An object managing the overall distributed system,
                                 providing global coordination (e.g., neighbour discovery).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []  # List to store (script, location) tuples assigned to this device
        
        # Events for synchronizing device operations
        self.timepoint_done = Event()  # Signals when all scripts for a timepoint are queued
        self.setup_done = Event()      # Signals when initial device setup is complete
        self.scripts_already_parsed = Event() # Signals that scripts for the current timepoint have been passed to workers
        
        self.queue = Queue()  # Queue to distribute scripts to worker threads
        self.barrier = None   # ReusableBarrier for synchronizing all devices at a timepoint
        self.location_locks = None # Dictionary to hold Locks for each data location, ensuring mutual exclusion
        self.neighbours = None # List of neighboring devices, retrieved from the supervisor
        
        # Create and start the main DeviceThread for this device
        self.thread = DeviceThread(self)
        self.thread.start()
        
        self.workers = []
        # Create and start 8 worker threads to process scripts concurrently
        for _ in xrange(8): # Loop 8 times to create 8 worker threads
            worker = Worker(self)
            worker.start()
            self.workers.append(worker)

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources (barrier, location_locks) across all devices.

        Device with device_id 0 acts as a coordinator, initializing the barrier
        and global location locks, then distributes them to other devices.
        Other devices wait for this setup to complete.

        Args:
            devices (list): A list of all Device instances in the system.
        """
        # Block Logic: Coordinator device (device_id == 0) initializes shared resources.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices)) # Initialize a barrier for all devices
            self.location_locks = {} # Initialize a dictionary for location-specific locks

            # Distribute the initialized barrier and location_locks to all other devices
            for device in devices:
                device.location_locks = self.location_locks
                device.barrier = self.barrier
                device.setup_done.set() # Signal that setup is done for this device
        else:
            # Other devices wait until the coordinator has completed the setup.
            self.setup_done.wait()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific data location on this device.

        If a script is provided, it's added to the internal list of scripts.
        If location-specific lock doesn't exist, it is created.
        If the device is already processing scripts for the current timepoint,
        the new script is immediately added to the queue for workers.
        If no script is provided (script is None), it signals the end of scripts
        for the current timepoint.

        Args:
            script (object): The script object to be executed.
            location (str): The identifier for the data location the script operates on.
        """
        if script is not None:
            # If a lock for this location does not exist, create it.
            if location not in self.location_locks:     
                self.location_locks[location] = Lock()  # Create a new Lock for this location
            self.scripts.append((script, location)) # Add the script and its location to the device's list
            
            # If scripts for the current timepoint are already being processed by workers,
            # immediately add this new script to the queue.
            if self.scripts_already_parsed.is_set():    
                self.queue.put((script, location))      # Place the script in the queue for workers
                                                        
                                                        
        else:
            # If script is None, it signifies that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set() # Signal that all scripts for the current timepoint are done being assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location.

        Args:
            location (str): The identifier of the data location.

        Returns:
            object: The sensor data at the given location, or None if the location is not found.
        """
        # Returns data if location exists in sensor_data, otherwise returns None.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location.

        Args:
            location (str): The identifier of the data location.
            data (object): The new data value to set for the location.
        """
        # Updates data only if the location already exists in sensor_data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates a graceful shutdown of the device's main thread and all worker threads.
        """
        self.thread.join() # Wait for the DeviceThread to complete its execution.
        
        # Send termination signals (None, None) to all worker threads.
        for i in xrange(8): # Assuming 8 worker threads
            self.queue.put((None, None))    # Place a sentinel value in the queue to signal worker shutdown.
                                            
        # Wait for all worker threads to complete their execution.
        for i in xrange(8): # Assuming 8 worker threads
            self.workers[i].join()


class DeviceThread(Thread):
    """
    The main thread responsible for managing the lifecycle of a Device.

    It orchestrates the processing of scripts in timepoints, handles communication
    with the supervisor to get neighbour information, and manages synchronization
    between script assignment and worker execution.
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
        1. Clears the 'scripts_already_parsed' event to prepare for a new timepoint.
        2. Fetches current neighbour information from the supervisor. If no neighbours
           (None returned), it signals the system shutdown.
        3. Adds all currently assigned scripts to the device's queue for worker processing.
        4. Sets the 'scripts_already_parsed' event to indicate scripts are ready for workers.
        5. Waits for the 'timepoint_done' event, signaling all scripts for the current timepoint
           have been processed.
        6. Clears 'timepoint_done' for the next cycle.
        7. Waits for the script queue to be empty, ensuring all tasks are done.
        8. Waits on the global barrier, synchronizing with other devices before proceeding.
        """
        while True: # Continuous loop for timepoint processing
            self.device.scripts_already_parsed.clear()      # Clear the event for a new cycle of script parsing.
                                                            
                                                            
            # Block Logic: Retrieve current neighbours and handle shutdown.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None: # Check for a shutdown signal from the supervisor
                break # Exit the loop and terminate the thread if no neighbours are returned.

            # Block Logic: Populate the queue with scripts for worker threads.
            # All scripts currently assigned to this device are put into the queue.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location))
            self.device.scripts_already_parsed.set()        # Signal that all scripts for this timepoint are in the queue.
                                                            
                                                            
                                                            
                                                            

            # Synchronization Point: Wait for all assigned scripts for the timepoint to be acknowledged.
            self.device.timepoint_done.wait() # Blocks until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.clear() # Reset the event for the next timepoint.
            
            # Synchronization Point: Wait for all worker threads to finish their current tasks.
            self.device.queue.join() # Blocks until all items in the queue have been processed and task_done() is called.
            
            # Global Synchronization Point: Wait for all devices to reach this point.
            self.device.barrier.wait() # Blocks until all devices have called wait() on the barrier.


class Worker(Thread):
    """
    A worker thread responsible for executing individual scripts assigned to a Device.

    Workers fetch scripts from the device's queue, acquire necessary locks for data
    integrity, execute the script with local and neighbor data, and update data.
    """

    def __init__(self, device):
        """
        Initializes a Worker thread.

        Args:
            device (Device): The Device instance this worker belongs to.
        """
        Thread.__init__(self, name="Worker")
        self.device = device

    def run(self):
        """
        The main execution loop for a Worker thread.

        This loop continuously:
        1. Fetches a script and its associated location from the device's queue.
        2. If a sentinel value (script is None) is received, the worker terminates.
        3. Acquires a location-specific lock to ensure exclusive access to shared data.
        4. Gathers relevant data from the current device and its neighbors for the location.
        5. Executes the script with the collected data.
        6. Updates the data on the current device and its neighbors with the script's result.
        7. Releases the location lock.
        8. Signals the device's queue that the task is done.
        """
        while True: # Continuous loop for processing scripts
            # Block Logic: Fetch script from queue and handle shutdown signal.
            (script, location) = self.device.queue.get() # Retrieve a script and its location from the queue.
            if script is None: # Check for a sentinel value to signal thread termination.
                break # Exit the loop and terminate the worker thread.

            # Critical Section: Data access protected by a location-specific lock.
            with self.device.location_locks[location]:  # Acquire the lock for the specific location.
                                                        
                                                        
                                                        
                                                        
                script_data = [] # List to accumulate data for script execution.
                
                # Block Logic: Gather data from neighboring devices.
                # Collects data from all neighbours at the specified location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Gather data from the current device.
                # Collects data from its own sensor data at the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Execute script if data is available.
                if script_data != []: # Ensures script is only run if there's data.
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Update data on neighboring devices.
                    # Propagates the script's result to all relevant neighbors.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Update data on the current device.
                    # Updates its own sensor data with the script's result.
                    self.device.set_data(location, result)

            # Signal that the current script task is complete for the queue.
            self.device.queue.task_done()

