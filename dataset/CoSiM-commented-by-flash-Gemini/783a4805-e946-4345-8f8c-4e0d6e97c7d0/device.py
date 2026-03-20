

"""
This module simulates a distributed system composed of interconnected devices.
It defines core components for device management, script execution, and synchronization.

Classes:
- `Device`: Represents an individual device in the distributed system, managing its state,
            sensor data, and interactions with other devices and a central supervisor.
- `RunScript`: A thread responsible for executing a specific script at a given location,
               collecting data from neighboring devices, and updating sensor data.
- `DeviceThread`: The main operational thread for each Device, handling its lifecycle,
                  script assignment, and synchronization with other devices using barriers.
"""

from threading import Event, Thread, Semaphore, Lock, RLock
from reusable_barrier import ReusableBarrier
import multiprocessing

class Device(object):
    """
    Represents a single computational device in a distributed simulation.

    Each device has a unique ID, sensor data, and interacts with a supervisor
    and other devices for synchronization and script execution. It manages
    its own thread of execution (`DeviceThread`) and uses various threading
    primitives for concurrency control.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing the sensor data managed by this device,
                                where keys are locations and values are data points.
            supervisor (Supervisor): A reference to the central supervisor managing the devices.
        """
        
        self.results = {}
        self.lock = None 
        self.dislocksdict = None 
        self.barrier = None
        self.sem = Semaphore(1)
        self.sem2 = Semaphore(0)
        self.all_devices = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
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
        Configures the current device and other devices in the system for synchronized operation.
        This method is critical for initializing shared synchronization primitives (like barriers and locks)
        across all devices. Only device_id 0 performs the initial setup of shared objects.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        loc = []
        # Block Logic: Gathers all unique sensor data locations across all devices.
        # Invariant: 'loc' temporarily stores all location keys from all device's sensor_data.
        for d in devices:
            for l in d.sensor_data:
                loc.append(l) 
        all_devices = devices
        # Block Logic: Initializes shared synchronization primitives only once by the device with ID 0.
        # Pre-condition: This block executes if the current device is the designated initializer (device_id == 0).
        # Post-condition: Shared barrier, distributed locks, and a global lock are initialized.
        if self.device_id == 0:
            self.sem2.release() # Releases a semaphore to allow other devices to proceed after initialization.
            self.barrier = ReusableBarrier(len(devices)) # Initializes a reusable barrier for all devices.
            self.dislocksdict = {} # Initializes a dictionary to hold RLock objects for each unique location.
            for k in list(set(loc)): # Block Logic: Creates a re-entrant lock for each unique sensor data location.
                self.dislocksdict[k] = RLock()
            self.lock = Lock() # Initializes a general-purpose lock for the device.

        self.sem2.acquire() # Block Logic: Waits for the initial setup to complete if not device_id 0.

        # Block Logic: Propagates the shared barrier, distributed locks, and global lock to all devices that haven't received them.
        # Pre-condition: 'devices' list contains all device objects in the system.
        for d in devices:
            if d.barrier == None: # Check if the device's barrier is not yet set.
                d.barrier = self.barrier # Assign the shared barrier.
                d.sem2.release() # Release semaphore to allow device to proceed.
                d.dislocksdict = self.dislocksdict # Assign the shared dictionary of distributed locks.
                d.lock = Lock() # Assign a general-purpose lock.

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a specific location.
        If 'script' is None, it signals that the current timepoint's script assignment is complete.

        Args:
            script (object or None): The script object to assign, or None to signal completion.
            location (str): The sensor data location to which the script pertains.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
   
    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from this device.

        Args:
            location (str): The identifier of the sensor data location.

        Returns:
            Any or None: The data associated with the location if present, otherwise None.
        """
        
        data = -1
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a specified location on this device.

        Args:
            location (str): The identifier of the sensor data location to update.
            data (Any): The new data value to set for the location.
        """
        
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by joining its associated thread.
        This ensures that the device's thread completes its execution before the program exits.
        """
        
        self.thread.join()

class RunScript(Thread):
    """
    A thread that executes a given script at a specific location for a device.
    It collects relevant sensor data from neighboring devices and the device itself,
    runs the script with this data, and then propagates the result back to
    the device and its neighbors. It uses RLock for distributed data access.
    """
    def __init__(self, script, location, neighbours, device):
        """
        Initializes a RunScript thread.

        Args:
            script (object): The script object to execute.
            location (str): The sensor data location relevant to this script.
            neighbours (list): A list of neighboring Device objects from which to fetch data.
            device (Device): The Device object owning this script execution.
        """
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device
    def run(self):
        """
        Executes the assigned script.
        This method acquires a distributed lock for the relevant location,
        gathers sensor data from the owner device and its neighbors for that location,
        runs the script with the collected data, and then propagates the result
        back to the owner device and its neighbors.
        """
        
        self.device.dislocksdict[self.location].acquire() # Acquire distributed lock for this location to ensure exclusive access.
        script_data = []
        # Block Logic: Gathers sensor data from neighboring devices.
        # Pre-condition: 'neighbours' is a list of Device objects.
        # Invariant: 'script_data' accumulates valid sensor data from neighbors.
        for device in self.neighbours:  
            device.lock.acquire() # Acquire local lock on neighbor device to safely read its data.
            data = device.get_data(self.location) 
            device.lock.release() # Release local lock on neighbor device.
            if data is not None:
                script_data.append(data)
                
        self.device.lock.acquire() # Acquire local lock on the owner device to safely read its data.
        data = self.device.get_data(self.location)
        self.device.lock.release() # Release local lock on the owner device.
        if data is not None:
            script_data.append(data)



        # Block Logic: Executes the script if sensor data was collected, then propagates the result.
        # Pre-condition: 'script_data' contains data points relevant to the script.
        if script_data != []:
            result = self.script.run(script_data) # Execute the script with the collected data.
            
            # Block Logic: Propagates the script's result to all neighboring devices.
            for device in self.neighbours:
                device.lock.acquire() # Acquire local lock on neighbor device to safely update its data.
                device.set_data(self.location, result)
                device.lock.release() # Release local lock on neighbor device.
            self.device.lock.acquire() # Acquire local lock on the owner device to safely update its data.
            self.device.set_data(self.location, result)
            self.device.lock.release() # Release local lock on the owner device.
        self.device.dislocksdict[self.location].release() # Release distributed lock for this location.


class DeviceThread(Thread):
    

    def __init__(self, device):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage and execute.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for the device thread.

        This loop continuously:
        1. Fetches neighbors from the supervisor. If no neighbors are returned (indicating shutdown),
           the loop breaks.
        2. Waits for a 'timepoint_done' event, signifying that scripts have been assigned for the current time step.
        3. Waits on a shared barrier to synchronize with other devices before executing scripts.
        4. Creates and starts `RunScript` threads for each assigned script.
        5. Joins all `RunScript` threads, waiting for their completion.
        6. Waits on the shared barrier again to synchronize after script execution.
        7. Clears the 'timepoint_done' event, preparing for the next time step.
        """
        while True:
            # Block Logic: Retrieves the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Block Logic: If no neighbors are returned (None), it signifies the simulation is ending.
                # Pre-condition: 'neighbours' being None is the termination signal from the supervisor.
                break
            # Block Logic: Waits for a signal that scripts have been assigned for the current timepoint.
            # Invariant: The device pauses here until `timepoint_done.set()` is called by the supervisor.
            self.device.timepoint_done.wait() 


            # Block Logic: Synchronizes all devices before script execution begins for the timepoint.
            self.device.barrier.wait() 
            script_threads = []
            # Block Logic: Creates and stores a `RunScript` thread for each assigned script.
            for (script, location) in self.device.scripts:
                script_threads.append(RunScript(script, location, neighbours, self.device))
            # Block Logic: Starts all `RunScript` threads concurrently.
            for t in script_threads:
                t.start() 
            # Block Logic: Waits for all `RunScript` threads to complete their execution.
            for t in script_threads:
                t.join() 
            # Block Logic: Synchronizes all devices again after all scripts have finished executing.
            self.device.barrier.wait()
            # Block Logic: Resets the `timepoint_done` event, preparing for the next timepoint's script assignment.
            self.device.timepoint_done.clear()
