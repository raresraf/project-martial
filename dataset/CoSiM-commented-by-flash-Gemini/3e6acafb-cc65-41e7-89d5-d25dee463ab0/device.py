

"""
@file device.py
@brief This module defines the architecture for a simulated device environment, featuring
       a master-worker pattern for concurrent script execution and synchronization.

@details It integrates a `ReusableBarrier` (imported), a `Queue` for script distribution,
         and a list of `Semaphore` objects for controlling access to data locations.
         The `Device` class represents individual simulation entities, `DeviceThread`
         manages a pool of `WorkerThread`s, and `WorkerThread`s execute scripts
         in parallel, coordinating via semaphores and location-specific locking.
         Device 0 acts as the leader, initializing shared synchronization primitives.
"""

from threading import Semaphore, Lock, Event, Thread
from Queue import Queue
from barrier import ReusableBarrier

class Device(object):
    """
    @brief Represents a single simulated device in the environment.

    @details Each device manages its own sensor data, interacts with a central supervisor,
             and executes scripts. It coordinates its activities through a dedicated
             `DeviceThread`, which in turn manages `WorkerThread`s for script execution.
             Shared `ReusableBarrier` and location-specific `Semaphore` objects are
             used for synchronization and data consistency during concurrent operations.
             Device 0 typically initializes these shared resources.
    """
    
    MAX_LOCATIONS = 100 # Maximum number of data locations, used for semaphore array size.
    CONST_ONE = 1       # Constant value, possibly for initialization or simple counts.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id An integer representing the unique identifier for the device.
        @param sensor_data A dictionary containing initial sensor data for the device.
        @param supervisor An object responsible for overseeing and coordinating devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to store (script, location) tuples assigned to this device.
        self.timepoint_done = Event() # Event to signal when all scripts for a timepoint are assigned.
        self.semaphore = [] # Shared list of semaphores for location-specific locking.
        self.thread = DeviceThread(self) # The dedicated thread for this device's main loop.
        # Initialize with a dummy barrier, will be replaced by the shared one during setup.
        self.barrier = ReusableBarrier(Device.CONST_ONE) 

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d" % self.device_id.
        """
        
        return "Device %d" % self.device_id

    def get_distributed_objs(self, barrier, semaphore):
        """
        @brief Receives and sets the globally shared synchronization objects.

        @details This method is called by the leader device (device 0) to distribute
                 the initialized `ReusableBarrier` and the list of location `Semaphore`s
                 to this device. It also starts the device's main thread.
        @param barrier An instance of `ReusableBarrier` shared across all devices.
        @param semaphore A list of `Semaphore` objects for location-specific locking.
        """

        self.barrier = barrier # Set the shared barrier.
        self.semaphore = semaphore # Set the shared list of semaphores.
        self.thread.start() # Start the main device thread.

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization resources (barrier and semaphores).

        @details This method is typically called once at the beginning of the simulation.
                 Only Device 0 (the leader) initializes the global `ReusableBarrier` and
                 a list of `Semaphore` objects, one for each potential data location.
                 These shared resources are then distributed to all other devices.
        @param devices A list of all Device objects in the simulation.
        """
        
        # Precondition: Only device with ID 0 is responsible for initializing shared resources.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices)) # Initialize the global reusable barrier.
            semaphore = [] # Initialize the list of semaphores.
            i = Device.MAX_LOCATIONS
            # Block Logic: Create a semaphore for each potential data location.
            while i > 0:
                semaphore.append(Semaphore(value=Device.CONST_ONE)) # Each semaphore has an initial value of 1 (binary semaphore acting as a lock).
                i = i - 1


            # Block Logic: Distribute the initialized barrier and semaphores to all devices.
            for device in devices:
                device.get_distributed_objs(barrier, semaphore)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.

        @details If a script is provided, it's added to the device's script queue.
                 If `script` is None, it signals the `timepoint_done` event,
                 indicating that all scripts for the current timepoint have been
                 assigned to this device.
        @param script The script object to assign, or None.
        @param location The data location (e.g., sensor ID) where the script will operate.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its target location.
        else:
            self.timepoint_done.set() # Signal that all script assignments for this timepoint are done.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location from the device's internal state.

        @param location The data location (e.g., sensor ID).
        @return The sensor data for the location, or None if the location is not found.
        """
        
        if location in self.sensor_data:
            obj = self.sensor_data[location]
        else:
            obj = None
        return obj

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location in the device's internal state.

        @param location The data location (e.g., sensor ID).
        @param data The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's dedicated thread.

        @details Joins the `DeviceThread`, ensuring all ongoing operations are completed
                 before the device fully shuts down.
        """
        
        self.thread.join() # Wait for the main device thread to finish its execution.



class DeviceThread(Thread):
    """
    @brief Manages the main operational loop for a Device object, coordinating script execution.

    @details This thread is responsible for the overall lifecycle of a device's operations
             within a timepoint. It fetches neighbor information from the supervisor,
             manages a pool of `WorkerThread`s, and distributes assigned scripts
             to them via a `Queue`. It also handles global synchronization using
             the shared `ReusableBarrier`.
    """
    
    THREADS_TO_START = 8  # Number of worker threads to spawn for concurrent script execution.
    MAX_SCRIPTS = 100     # Maximum number of scripts that can be queued at once.

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device The `Device` object that this thread is responsible for.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device # Reference to the parent Device object.
        self.threads = []    # List to hold references to WorkerThread instances.
        self.neighbours = [] # List to store neighboring devices for data interaction.
        self.queue = Queue(maxsize=DeviceThread.MAX_SCRIPTS) # Queue for scripts to be processed by worker threads.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        @details This method sets up a pool of `WorkerThread`s and then continuously
                 processes timepoints in the simulation. For each timepoint, it
                 fetches neighbor information, waits for scripts to be assigned,
                 puts scripts into a queue for worker threads, waits for workers
                 to complete, and finally synchronizes globally using the shared barrier.
                 The loop terminates when the supervisor signals the end of the simulation.
        """
        lock = Lock() # Lock for protecting access to the script queue.
        # Block Logic: Spawn a pool of WorkerThreads.
        for i in range(DeviceThread.THREADS_TO_START):
            self.threads.append(WorkerThread(self, i, self.device, self.queue, lock))
            self.threads[i].setDaemon(True) # Set as daemon to allow main program to exit without waiting for them.
        for thread in self.threads:
            thread.start() # Start all worker threads.

        while True:
            # Block Logic: Query the supervisor for updated neighbor information for the current timepoint.
            self.neighbours = self.device.supervisor.get_neighbours()
            # Precondition: If supervisor returns None, it signals the end of the simulation.
            if self.neighbours is None:
                break # Exit the main loop.

            # Block Logic: Wait for the Device to signal that all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Populate the queue with scripts to be processed by worker threads.
            for script in self.device.scripts:
                self.queue.put(script) # Add script (tuple of script object and location) to the queue.
            self.queue.join() # Wait until all items in the queue have been processed by worker threads.

            self.device.timepoint_done.clear() # Reset the event for the next timepoint.

            # Global synchronization barrier to ensure all devices complete their timepoint processing.
            self.device.barrier.wait()

        # Block Logic: Clean up by joining worker threads (though set as daemon, explicit join for robustness).
        for thread in self.threads:
            thread.join()


class WorkerThread(Thread):
    """
    @brief An auxiliary thread responsible for processing scripts from a shared queue.

    @details These threads are part of a pool managed by `DeviceThread`. Each worker
             acquires a script from the queue, obtains a location-specific semaphore,
             gathers data from neighbors, executes the script, updates results,
             and then releases the semaphore and marks the script as done in the queue.
    """
    
    def __init__(self, master, worker_id, device, queue, lock):
        """
        @brief Initializes a new WorkerThread instance.

        @param master The parent `DeviceThread` managing this worker.
        @param worker_id An integer representing the unique identifier for this worker thread.
        @param device The parent `Device` object.
        @param queue The shared `Queue` from which scripts are pulled.
        @param lock A `threading.Lock` to protect access to the shared queue's empty check.
        """
        
        Thread.__init__(self, name="Worker Thread %d %d" % (worker_id, device.device_id))
        self.master = master # Reference to the controlling DeviceThread.
        self.device = device # Reference to the parent Device object.
        self.queue = queue   # The shared script queue.
        self.lock = lock     # Lock for queue access.

    def run(self):
        """
        @brief The main execution logic for the WorkerThread.

        @details This method continuously attempts to retrieve scripts from the shared
                 queue. If a script is available, it acquires a location-specific semaphore,
                 gathers data, executes the script, propagates results, and then
                 releases the semaphore. It marks the task as done in the queue.
                 The loop breaks when the master `DeviceThread` signals the end
                 of the simulation.
        """

        while True:
            # Block Logic: Acquire a lock to safely check if the queue is empty and get a script.
            self.lock.acquire()
            value = self.queue.empty() # Check if queue is empty.
            if value is False:
                (script, location) = self.queue.get() # Get a script from the queue.
            self.lock.release() # Release the queue access lock.

            # Precondition: Process the script only if one was retrieved from the queue.
            if value is False:
                script_data = [] # List to accumulate data for the script.

                # Inline: Acquire the semaphore for the specific data location to ensure exclusive access.
                self.device.semaphore[location].acquire()

                # Block Logic: Gather data from neighboring devices for the current location.
                for device in self.master.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Get data from the current device for the current location.
                data = self.master.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Precondition: Execute the script only if there is data available.
                if script_data != []:
                    # Execute the assigned script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Update sensor data on neighboring devices.
                    for device in self.master.neighbours:
                        device.set_data(location, result)
                    
                    # Update sensor data on the current device.
                    self.master.device.set_data(location, result)

                # Inline: Release the semaphore for the data location.
                self.device.semaphore[location].release()
                self.queue.task_done() # Signal that the script has been processed.

            # Precondition: Check if the master DeviceThread has signaled the end of the simulation.
            if self.master.neighbours is None:
                break # Exit the worker thread's loop.

