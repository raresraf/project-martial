

"""
This module simulates a distributed system composed of interconnected devices,
utilizing a reusable barrier and a complex script execution model with fine-grained locking.

Classes:
- `Device`: Represents an individual device in the distributed system, managing its state,
            sensor data, and interactions with other devices and a central supervisor.
- `DeviceThread`: The main operational thread for each Device, handling its lifecycle,
                  script assignment, data collection, and synchronized script execution.

Functions:
- `run_script`: A helper function executed by a new thread, responsible for gathering data
                for a specific script, executing it, and propagating results.
"""

from threading import Event, Thread, Lock, Semaphore
from cond_barrier import ReusableBarrier


class Device(object):
    """
    Represents a single computational device in a distributed simulation.

    Each device has a unique ID, manages its own sensor data, and interacts
    with a central supervisor for coordination. It uses a `DeviceThread` for
    its main operations and employs various locks and a shared barrier for
    thread synchronization and safe access to shared resources.
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
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.scripts_to_run = []
        self.timepoint_done = Event()
        self.setup_done = Event()
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
        Configures shared synchronization primitives (barrier and locks) across all devices.
        This method ensures that the barrier and specific locks for neighbors and locations
        are either initialized by device_id 0 or copied from device_id 0 to others.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        
        nr_devices = len(devices)
        # Block Logic: Initializes shared synchronization primitives.
        # This section ensures that the barrier and various locks are correctly
        # set up and shared among all devices. Device with ID 0 acts as the initializer.
        
        # Invariant: After this block, all devices will share the same barrier and lock objects.
        
        # If this device is device_id 0, it initializes the shared objects.
        # Otherwise, it copies the references from device 0.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(nr_devices) # Initializes a reusable barrier for all devices.
            self.lock_get_neigh = Lock() # Initializes a lock for accessing neighbors data.
            self.lock_location = {} # Dictionary to hold locks per location.
            self.lock_check_loc = Lock() # Lock for checking location locks.
            self.lock_scripts = Lock() # Lock for managing assigned scripts.
        else:
            # Block Logic: Copies shared synchronization primitives from device 0.
            # Pre-condition: Device with ID 0 has already initialized the shared objects.
            # Invariant: This device now holds references to the shared barrier and locks.
            for device in devices: # Iterates to find device with ID 0.
                if device.device_id == 0:
                    self.barrier = device.barrier
                    self.lock_get_neigh = device.lock_get_neigh
                    self.lock_location = device.lock_location
                    self.lock_check_loc = device.lock_check_loc
                    self.lock_scripts = device.lock_scripts
                    break # Break once device 0 is found and its shared objects are copied.

        # Block Logic: Initializes a lock for each sensor data location if not already present.
        # Pre-condition: `self.sensor_data` contains keys for all managed locations.
        # Invariant: Each location in `self.sensor_data` has an associated lock in `self.lock_location`.
        for location in self.sensor_data:
            if not self.lock_location.has_key(location):
                self.lock_location[location] = Lock()

        # Signals that this device has completed its setup.
        self.setup_done.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device at a specific location.
        If 'script' is None, it signals that the current timepoint's script assignment is complete.

        Args:
            script (object or None): The script object to assign, or None to signal completion.
            location (str): The sensor data location to which the script pertains.
        """
        
        if script is not None:
            self.lock_scripts.acquire()
            self.scripts.append((script, location))
            self.scripts_to_run.append((script, location))
            self.lock_scripts.release()
        else:
            self.lock_scripts.acquire()
            self.timepoint_done.set()
            self.lock_scripts.release()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from this device.

        Args:
            location (str): The identifier of the sensor data location.

        Returns:
            Any or None: The data associated with the location if present, otherwise None.
        """
        
        return (self.sensor_data[location]
                if location in self.sensor_data else None)

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


class DeviceThread(Thread):
    """
    The main thread of execution for a single Device object.

    This thread is responsible for the continuous operation of the device,
    including fetching neighbors, managing script execution in a synchronized
    manner (using semaphores and locks), and coordinating with other devices
    through a shared barrier.
    """
    

    def __init__(self, device):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage and execute.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.sem_threads = Semaphore(8) 

    def run(self):
        """
        Main execution loop for the device thread.

        This loop continuously:
        1. Waits for initial setup to complete (`self.device.setup_done.wait()`).
        2. Fetches neighbors from the supervisor. If no neighbors are returned (indicating shutdown),
           the loop breaks.
        3. Manages script execution for assigned scripts, handling concurrency using semaphores and locks.
           Scripts are executed in separate threads, and results are propagated back to the devices.
        4. Synchronizes with other devices using a shared barrier before proceeding to the next timepoint.
        """
        self.device.setup_done.wait() # Waits until all devices have completed their initial setup.

        while True:
            threads = [] # List to hold threads executing scripts.

            # Block Logic: Acquires a lock before fetching neighbors from the supervisor to ensure thread safety.
            self.device.lock_get_neigh.acquire()
            neighbours = self.device.supervisor.get_neighbours() # Fetches the current set of neighboring devices.
            self.device.lock_get_neigh.release()

            if neighbours is None:
                # Block Logic: If no neighbors are returned (None), it signals that the simulation is ending.
                # Post-condition: The device thread terminates its execution.
                break

            # Block Logic: Acquires a lock to safely copy the list of scripts to run for the current timepoint.
            self.device.lock_scripts.acquire()
            self.device.scripts_to_run = self.device.scripts[:] # Creates a shallow copy of assigned scripts.
            # Block Logic: Checks if all scripts for the current timepoint have been processed and if the timepoint is done.
            finished = (self.device.timepoint_done.is_set() and
                        len(self.device.scripts_to_run) == 0)
            self.device.lock_scripts.release()

            # Block Logic: Loop to continuously attempt to run scripts until all assigned scripts are finished.
            # Invariant: Scripts are processed and removed from `scripts_to_run` until the list is empty
            #            and `timepoint_done` is set, indicating all scripts for the current timepoint are done.
            while not finished:
                # Block Logic: Acquires a lock to safely get the current list of scripts to run.
                self.device.lock_scripts.acquire()
                local_scripts_to_run = self.device.scripts_to_run[:] # Create a local copy to iterate safely.
                self.device.lock_scripts.release()

                # Block Logic: Iterates through scripts, attempting to execute them if their location is not locked.
                for (script, location) in local_scripts_to_run:
                    
                    self.sem_threads.acquire() # Acquire a semaphore permit, limiting concurrent script execution.

                    # Block Logic: Acquires a lock to safely check the status of the location-specific lock.
                    self.device.lock_check_loc.acquire()

                    if self.device.lock_location[location].locked(): # Check if the lock for the current location is already held.
                        self.device.lock_check_loc.release() # Release check lock.
                        self.sem_threads.release() # Release semaphore permit.
                        continue # Skip to the next script if the location is locked.

                    self.device.lock_location[location].acquire() # Acquire the lock for the current location.

                    # Block Logic: Acquires a lock to safely remove the script from the list of scripts to run.
                    self.device.lock_scripts.acquire()
                    self.device.scripts_to_run.remove((script, location)) # Remove the script as it's being processed.
                    self.device.lock_scripts.release()

                    self.device.lock_check_loc.release() # Release check lock.

                    # Block Logic: Creates and starts a new thread to execute the `run_script` function.
                    thread = Thread(target=run_script, args=(self, neighbours,
                                                             script, location))
                    threads.append(thread)
                    thread.start()

                # Block Logic: Re-evaluates the 'finished' condition after attempting to launch scripts.
                self.device.lock_scripts.acquire()
                finished = (self.device.timepoint_done.is_set() and
                            len(self.device.scripts_to_run) == 0)
                self.device.lock_scripts.release()

            # Block Logic: Waits for all launched script execution threads to complete.
            for thread in threads:
                thread.join()

            # Block Logic: Clears the 'timepoint_done' event, preparing for the next timepoint.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes all devices using the shared barrier before proceeding to the next iteration.
            self.device.barrier.wait()

def run_script(parent_device_thread, neighbours, script, location):
    """
    Helper function to execute a script in a separate thread.

    This function is responsible for:
    1. Collecting relevant sensor data for a specific `location` from `parent_device_thread.device`
       and its `neighbours`.
    2. Executing the provided `script` with the collected data.
    3. Propagating the `result` of the script execution back to the `parent_device_thread.device`
       and its `neighbours` at the specified `location`.
    4. Releasing the location lock and semaphore permit upon completion.

    Args:
        parent_device_thread (DeviceThread): The `DeviceThread` instance that spawned this execution.
        neighbours (list): A list of neighboring Device objects.
        script (object): The script object to execute, which must have a `run` method.
        location (str): The sensor data location relevant to this script.
    """
    
    script_data = []
    
    # Block Logic: Collects sensor data from each neighboring device for the current location.
    # Pre-condition: `neighbours` is a list of Device objects.
    # Invariant: `script_data` accumulates valid sensor data from neighbors.
    for device in neighbours:
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)
    
    # Block Logic: Collects sensor data from the parent device for the current location.
    data = parent_device_thread.device.get_data(location)
    if data is not None:
        script_data.append(data)

    if script_data != []:
        # Block Logic: Executes the script with the collected data if data is available.
        result = script.run(script_data)

        # Block Logic: Propagates the script's result to all neighboring devices.
        for device in neighbours:
            device.set_data(location, result)
        
        # Block Logic: Updates the sensor data on the parent device with the script's result.
        parent_device_thread.device.set_data(location, result)

    # Block Logic: Releases the location-specific lock and a semaphore permit upon completion.
    parent_device_thread.device.lock_location[location].release()
    parent_device_thread.sem_threads.release()
