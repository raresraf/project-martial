


from threading import Event, Thread, Semaphore, Lock, RLock
from reusable_barrier import ReusableBarrier
import multiprocessing

"""
This module implements a simulation framework for distributed devices,
focusing on concurrent execution of scripts and synchronized data processing.
It utilizes a `Device` class to represent each simulated entity,
a `DeviceThread` for main control flow, and `RunScript` threads for
executing individual scripts. A distributed locking mechanism is employed
to ensure consistent data access across multiple devices.
"""


from threading import Event, Thread, Semaphore, Lock, RLock
from reusable_barrier import ReusableBarrier # External dependency: Assumes ReusableBarrier is defined elsewhere.
import multiprocessing # Imported but not directly used in the provided snippet.

class Device(object):
    """
    Represents a simulated device within a distributed system.

    Each device manages its sensor data, interacts with a central supervisor,
    and dispatches scripts for execution. It launches a dedicated `DeviceThread`
    to handle control flow and participates in global synchronization.
    A sophisticated distributed locking mechanism is used for data locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor readings
                                (e.g., {location_id: data_value}).
            supervisor (object): An object representing the central supervisor,
                                 used for coordination (e.g., getting neighbors).
        """
        # Dictionary to store script execution results (unused in provided snippet).
        self.results = {}
        # Local lock for protecting access to this device's `sensor_data` (acquired by `RunScript` for local data access).
        self.lock = None
        # Dictionary of reentrant locks (`RLock`), shared across all devices, with one lock per data location.
        # This provides distributed exclusive access control to specific data points.
        self.dislocksdict = None
        # Reference to the global barrier used for synchronizing all devices.
        self.barrier = None
        # Semaphore used for general synchronization, initialized with a count of 1.
        self.sem = Semaphore(1)
        # Semaphore used specifically for coordinating the setup phase, initialized with a count of 0.
        self.sem2 = Semaphore(0)
        # List to store references to all devices in the system (unused in provided snippet).
        self.all_devices = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a script has been received (usage unclear from `assign_script` logic).
        self.script_received = Event()
        # List to temporarily store scripts assigned to this device before dispatching.
        self.scripts = []
        # Event to signal when the current timepoint's script assignments are done.
        self.timepoint_done = Event()
        # The main thread for the device, responsible for supervisor interaction and dispatching scripts.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global synchronization primitives (barrier and distributed data locks)
        and distributes them among all devices.

        This method is typically called once during the initialization phase of the simulation
        and uses a semaphore to ensure ordered setup, with device 0 acting as the coordinator.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        loc = [] # Temporary list to collect all unique data locations across all devices.
        # Block Logic: Gathers all unique data locations from all devices.
        for d in devices:
            for l in d.sensor_data:
                loc.append(l)
        all_devices = devices # Stores a reference to all devices (unused in provided snippet).
        # Block Logic: Only device with ID 0 acts as the coordinator for global setup.
        if self.device_id == 0:
            self.sem2.release() # Releases `sem2` to allow other devices to proceed with setup.
            # Functional Utility: Creates a global barrier that all devices will use for synchronization.
            self.barrier = ReusableBarrier(len(devices))
            # Functional Utility: Initializes a dictionary of reentrant locks (`RLock`),
            # one for each unique data location across the entire system.
            self.dislocksdict = {}
            for k in list(set(loc)): # Ensures each unique location gets one RLock.
                self.dislocksdict[k] = RLock()
            # Initializes this device's local lock for its sensor_data.
            self.lock = Lock()

        # Block Logic: Waits for device 0 to complete its initial setup coordination.
        # This ensures `barrier` and `dislocksdict` are initialized before other devices access them.
        self.sem2.acquire()

        # Block Logic: Distributes the globally initialized `barrier` and `dislocksdict`
        # to all other devices that haven't received them yet. Also initializes their local locks.
        for d in devices:
            if d.barrier == None: # Checks if the device's barrier has not been set yet.
                d.barrier = self.barrier # Assigns the global barrier.
                d.sem2.release() # Releases `sem2` for the next device in sequence.
                d.dislocksdict = self.dislocksdict # Assigns the global dictionary of location locks.
                d.lock = Lock() # Initializes the device's local lock.

    def assign_script(self, script, location):
        """
        Assigns a script for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If script is None, it signifies the end of script assignments for the current timepoint.
            # Functional Utility: Signals that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()
   
    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Note: This method does NOT acquire the device's local lock (`self.lock`)
        or the distributed location lock (`self.dislocksdict[location]`).
        Locking for data access is expected to be handled by the caller (`RunScript`).

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or -1 if not found.
                 (Returns `None` if `location` not in `sensor_data`.)
        """
        data = -1 # Default return value if location is found but data is -1.
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None # Returns None if the location is not present in sensor_data.

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Note: This method does NOT acquire the device's local lock (`self.lock`)
        or the distributed location lock (`self.dislocksdict[location]`).
        Locking for data access is expected to be handled by the caller (`RunScript`).

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Performs a graceful shutdown of the main device thread.
        """
        # Waits for the main device thread (`DeviceThread`) to complete its execution.
        self.thread.join()

class RunScript(Thread):
    """
    A worker thread responsible for executing a single script at a specific data location.

    This thread collects data from its associated device and its neighbors,
    executes the provided script, and updates the relevant sensor data.
    It utilizes a combination of distributed (location-specific) and local (device-specific)
    locks to ensure data consistency during concurrent access.
    """
    def __init__(self, script, location, neighbours, device):
        """
        Initializes a RunScript thread.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location the script operates on.
            neighbours (list): A list of neighboring Device objects.
            device (Device): The `Device` instance this thread is associated with.
        """
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        Main execution logic for the RunScript thread.

        Architectural Intent: Ensures thread-safe access to a specific data location
        across devices. It collects data, executes the assigned script, and updates
        the result on all relevant devices (self and neighbors) using a combination
        of distributed and local locking mechanisms.
        """
        # Functional Utility: Acquires the distributed (location-specific) reentrant lock.
        # This ensures exclusive access to the data at `self.location` across all devices.
        self.device.dislocksdict[self.location].acquire()
        script_data = [] # List to store all data relevant to the script.

        # Block Logic: Collects sensor data from neighboring devices.
        for device in self.neighbours:
            # Functional Utility: Acquires the neighbor's local lock before accessing its data.
            device.lock.acquire()
            data = device.get_data(self.location)
            device.lock.release() # Releases the neighbor's local lock.
            if data is not None:
                script_data.append(data) # Adds neighbor's data if available.
                
        # Block Logic: Collects sensor data from its own device.
        self.device.lock.acquire() # Acquires its own device's local lock.
        data = self.device.get_data(self.location)
        self.device.lock.release() # Releases its own device's local lock.
        if data is not None:
            script_data.append(data) # Adds local device data if available.

        # Block Logic: If any data was collected, executes the script and updates data.
        if script_data != []:
            # Functional Utility: Executes the assigned script with the collected data.
            result = self.script.run(script_data)
            
            # Block Logic: Updates data on neighboring devices.
            for device in self.neighbours:
                device.lock.acquire() # Acquires the neighbor's local lock before updating its data.
                device.set_data(self.location, result)
                device.lock.release() # Releases the neighbor's local lock.
            # Block Logic: Updates data on its own device.
            self.device.lock.acquire() # Acquires its own device's local lock before updating its data.
            self.device.set_data(self.location, result)
            self.device.lock.release() # Releases its own device's local lock.
        self.device.dislocksdict[self.location].release() # Releases the distributed location lock.


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for coordinating with the supervisor,
    dispatching scripts to `RunScript` workers, and participating in global synchronization.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The `Device` instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for the DeviceThread.

        Architectural Intent: Continuously fetches neighbor information from the supervisor.
        It orchestrates the execution of assigned scripts by launching a new `RunScript`
        for each, waits for them to complete, and then participates in a global barrier
        for timepoint synchronization.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            # Termination Condition: If no neighbors are returned (None), it signifies
            # that the simulation is ending for this device, and the loop breaks.
            if neighbours is None:
                break
            # Block Logic: Waits until the current timepoint's script assignments are done.
            self.device.timepoint_done.wait()

            # Functional Utility: Participates in the global barrier. This likely ensures
            # all devices have received their scripts and are ready to start processing.
            self.device.barrier.wait()
            script_threads = [] # List to keep track of active RunScript instances.
            # Block Logic: For each assigned script, a new RunScript thread is created.
            for (script, location) in self.device.scripts:
                script_threads.append(RunScript(script, location, neighbours, self.device))
            # Block Logic: Starts all created RunScript threads concurrently.
            for t in script_threads:
                t.start()
            # Block Logic: Waits for all launched RunScript instances to complete their execution.
            for t in script_threads:
                t.join()
            # Functional Utility: Participates in the global barrier again. This likely ensures
            # all devices have finished executing their scripts before proceeding to the next timepoint.
            self.device.barrier.wait()
            # Functional Utility: Clears the event to prepare for the next timepoint.
            self.device.timepoint_done.clear()
