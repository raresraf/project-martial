


from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a simulated device within a multi-device system.

    This device manages its own sensor data, processes assigned scripts through
    a pool of worker threads, and synchronizes its operations with other devices
    using a shared barrier and location-specific semaphores for resource control.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor readings,
                                 where keys are locations and values are data.
            supervisor (Supervisor): A reference to the supervisor object
                                     that manages device interactions.
        """
        self.device_id = device_id
        self.read_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that scripts have been assigned for the current timepoint.
        self.script_received = Event()
        # List to store scripts assigned to this device for processing.
        self.scripts = []
        # The main thread responsible for managing this device's operations.
        self.thread = DeviceThread(self)
        # Shared barrier for synchronizing all devices at the end of each processing round.
        self.loopBarrier = None
        # Shared dictionary to store semaphores for individual locations, preventing race conditions.
        self.locationSemaphores = None
        # Start the device's main thread upon initialization.
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization mechanisms for all devices in the system.

        This method initializes a `ReusableBarrierCond` for inter-device synchronization
        and a shared dictionary for `locationSemaphores`. These are then distributed
        to all `Device` instances.

        Args:
            devices (list): A list of all `Device` instances in the system.
        """
        # Create a shared barrier for synchronizing all devices.
        loopBarrier = ReusableBarrierCond(len(devices))
        # Create a shared dictionary to hold semaphores for different locations.
        locationSemaphores = {}
        # Distribute the shared barrier and location semaphores to all devices.
        for device in devices :
            device.loopBarrier = loopBarrier
            device.locationSemaphores = locationSemaphores

    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device.

        If a script is provided, it's added to the device's list of scripts.
        If a semaphore for the script's location doesn't exist in the shared
        `locationSemaphores` dictionary, a new one is created.
        If `script` is None, it signals that script assignment for the current
        timepoint is complete, and the `script_received` event is set.

        Args:
            script (object or None): The script object to assign, or None to signal
                                     the end of script assignments for a timepoint.
            location (int): The identifier for the location associated with the script.
        """
        if script is not None:
            # Add the script and its location to the device's list.
            self.scripts.append((script, location))
            # If a semaphore for this location doesn't exist, create one.
            if self.locationSemaphores.get(location) is None:
                self.locationSemaphores[location] = Semaphore()
        else:
            # If script is None, signal that all scripts for this timepoint have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The identifier for the location for which to retrieve data.

        Returns:
            Any: The sensor data if the location exists in read_data, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (int): The identifier for the location where the data should be set.
            data (Any): The new data to set for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its associated thread.
        This ensures that the device's main thread completes its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the lifecycle and execution of ScriptThread instances for a Device.

    This thread orchestrates rounds of script processing: it continuously
    fetches neighbor information, waits for scripts to be assigned, dispatches
    these scripts to a pool of `ScriptThread`s for parallel execution,
    waits for their completion, and then synchronizes with other DeviceThreads
    using a shared barrier before starting the next round.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # A lock to control access to script assignment to worker threads.
        self.workLock = Lock()
        # Keeps track of the index of the last script given to a worker.
        self.lastScriptGiven = 0;

    def run(self):
        """
        The main execution loop for the DeviceThread.

        Each iteration represents a processing round. It performs the following steps:
        1. Retrieves up-to-date neighbor information from the supervisor.
        2. If no neighbors are returned (e.g., simulation end), the loop breaks.
        3. Resets the script distribution pointer (`lastScriptGiven`).
        4. Waits until scripts for the current timepoint have been assigned.
        5. Creates and starts a pool of `ScriptThread` instances (workers).
        6. Waits for all `ScriptThread` instances to complete their tasks.
        7. Clears the `script_received` event for the next round.
        8. Synchronizes with other DeviceThreads using the `loopBarrier`.
        """
        while True:
            # Get updated neighbor information from the supervisor for the current round.
            neighbours = self.device.supervisor.get_neighbours()
            # If supervisor returns None, it signals the simulation to terminate.
            if neighbours is None:
                break

            # Reset the counter for distributing scripts to workers for the new round.
            self.lastScriptGiven = 0;

            # Wait until the current set of scripts for this timepoint has been assigned.
            self.device.script_received.wait()
            
            # List to hold references to the active worker threads for the current round.
            workers = []
            # A local lock for the workers to control script retrieval from getWork().
            # (Note: workLock is already a member of DeviceThread, this local variable shadows it)
            workLock = Lock()


            # Create and populate a pool of ScriptThread workers. The number of workers
            # is limited to a maximum of 8 or the total number of scripts available.
            for i in range(0, min(8, len(self.device. scripts))):
                worker = ScriptThread(self, neighbours, workLock)
                workers.append(worker)

            # Start all worker threads concurrently.
            for worker in workers:
                worker.start()

            # Wait for all worker threads to complete their execution before proceeding.
            for worker in workers:
                worker.join()

            # Clear the event, indicating that scripts for this round have been processed.
            self.device.script_received.clear()

            # Wait at the shared barrier to synchronize with all other devices.
            self.device.loopBarrier.wait()

    def getWork(self):
        """
        Provides the next available script and its location to a worker thread.

        This method is thread-safe, ensuring that each script is handed out
        only once. It iterates through the device's assigned scripts.

        Returns:
            tuple or None: A tuple (script, location) if a script is available,
                           otherwise None if all scripts have been distributed.
        """
        script = None
        # Ensure exclusive access to lastScriptGiven counter.
        with self.workLock: # Using the instance's workLock.
            if (self.lastScriptGiven < len(self.device.scripts)):
                script = self.device.scripts[self.lastScriptGiven]
                self.lastScriptGiven += 1

        return script

class ScriptThread(Thread) :
    """
    A worker thread dedicated to executing a single script within a device's processing round.

    Each `ScriptThread` acquires a script and its associated location from the
    `DeviceThread` (master), then acquires a semaphore for that location to
    ensure exclusive access. It collects sensor data from the device and its
    neighbors, executes the script with this data, updates the sensor data,
    and finally releases the location semaphore.
    """
    
    def __init__(self, master, neighbours, workLock):
        """
        Initializes a new ScriptThread instance.

        Args:
            master (DeviceThread): The `DeviceThread` instance that created this worker.
            neighbours (list): A list of neighboring Device objects for data collection.
            workLock (Lock): A lock used by the master to control script distribution.
        """
        Thread.__init__(self)
        self.master = master
        self.neighbours = neighbours
        self.workLock = workLock # This workLock is redundant as master.workLock is used.

    def run(self) :
        """
        The main execution method for the ScriptThread.

        It continuously attempts to acquire a script from the `DeviceThread`.
        For each acquired script:
        1. Acquires the location-specific semaphore to prevent concurrent access to that location.
        2. Collects sensor data from itself and its neighbors for the given location.
        3. Executes the script with the collected data.
        4. Updates the sensor data on the device and its neighbors with the script's result.
        5. Releases the location-specific semaphore.
        The loop continues until no more scripts are available from the `DeviceThread`.
        """
        # Acquire the master's workLock to get a script and its location.
        # This prevents multiple ScriptThreads from taking the same script.
        self.master.workLock.acquire()
        scriptLocation = self.master.getWork()
        self.master.workLock.release()

        # Continue processing scripts until getWork() returns None.
        while scriptLocation is not None:
            (script, location) = scriptLocation
            script_data = [] # List to store data collected for the script.
            
            # Acquire the semaphore for the specific location. This ensures that only
            # one thread processes data for a given location at a time.
            self.master.device.locationSemaphores.get(location).acquire()
            # Collect data from neighboring devices for the current location.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Collect data from the current device itself for the current location.
            data = self.master.device.get_data(location)
            if data is not None:
                script_data.append(data)


            if script_data != []:
                # Execute the script with the collected data.
                result = script.run(script_data)

                # Update the data in neighboring devices.
                for device in self.neighbours:
                    device.set_data(location, result)
                # Update the data in the current device.
                self.master.device.set_data(location, result)

            # Release the semaphore for the current location.
            self.master.device.locationSemaphores.get(location).release()
            
            # Acquire the master's workLock again to get the next script.
            self.master.workLock.acquire()
            scriptLocation = self.master.getWork()
            self.master.workLock.release()