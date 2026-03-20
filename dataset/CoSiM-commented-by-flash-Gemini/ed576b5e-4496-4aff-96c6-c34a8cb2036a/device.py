"""
This module implements a distributed device system for sensor data processing,
featuring a reusable barrier for synchronization and a multi-threaded approach
to parallelize script execution within each device.

Core components include:
- Device: Represents a node in the distributed network, managing sensor data
          and coordinating with a supervisor and other devices.
- DeviceThread: The main thread for a Device, orchestrating script execution
                and inter-device communication.
- ExecutorThread: Worker threads responsible for executing individual scripts
                  and handling data exchange for specific locations.
"""

from threading import Event, Thread, Lock
from mybarrier import ReusableBarrier # Assumed to be a custom reusable barrier implementation.

class Device(object):
    """
    Represents an individual device within a distributed sensor network.
    Each device has a unique ID, local sensor data, and can interact with
    a central supervisor and other peer devices. It is capable of receiving
    and executing processing scripts.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping locations (str) to sensor data values.
            supervisor (Supervisor): A reference to the central supervisor object,
                                     used for network-wide queries (e.g., getting neighbors).
        
        Functional Utility: Sets up the device's unique state, communication channels,
                            and synchronization primitives needed for its operations.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a new script has been assigned to this device.
        self.scripts = [] # List to store assigned scripts (script, location) tuples.
        self.timepoint_done = Event() # Event to signal when the device has completed its processing for a timepoint.
        self.thread = DeviceThread(self) # The dedicated thread that runs the device's main logic.
        
        self.reusable_barrier = None # Reference to a shared ReusableBarrier for network-wide synchronization.
        self.devices = [] # List of all Device objects in the network.
        # List of Locks, where each lock protects access to a specific data location
        # across all devices. The size '100' seems arbitrary and might need
        # dynamic adjustment or a more robust mapping for production use.
        self.location_lock = [] 
        self.set_lock = Lock() # Lock protecting write access to sensor_data.
        self.get_lock = Lock() # Lock protecting read access to sensor_data.
        self.ready = Event() # Event to signal that the device's shared resources are set up.
                             
        self.thread.start() # Starts the Device's main processing thread.

    def __str__(self):
        """
        Returns a human-readable string representation of the Device.
        Functional Utility: Aids in identification and logging of device activities.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures shared network resources, such as the global barrier and
        location-specific locks. This method is typically called by a designated
        master device (device_id == 0) to ensure consistent initialization.

        Args:
            devices (list): A list of all Device instances in the network.
        
        Functional Utility: Initializes and distributes common synchronization
                            primitives used across all devices.
        Block Logic: If this device is the master (ID 0), it creates the
                     `ReusableBarrier` and a pool of `location_lock`s, then
                     propagates these shared instances to all other devices
                     and sets their `ready` event.
        Pre-condition: Called once by the master device after all devices are instantiated.
        """
        self.devices = devices
        if self.device_id == 0:
            reusable_barrier = ReusableBarrier(len(devices))
            # Initialize a fixed number of location locks.
            # The '100' here suggests a maximum number of distinct locations/data points.
            for _ in range(100): 
                self.location_lock.append(Lock())

            for dev in devices:
                dev.reusable_barrier = reusable_barrier
                dev.location_lock = self.location_lock
                dev.ready.set() # Signals to the DeviceThread that shared resources are ready.

    def assign_script(self, script, location):
        """
        Assigns a processing script to be executed for a specific data location.

        Args:
            script (Script): The script object containing the `run` method for processing.
            location (str): The identifier for the data location this script targets.
        
        Functional Utility: Adds new processing tasks to the device's queue.
        Block Logic: If a script is provided, it's appended to the `scripts` list.
                     If no script is provided (i.e., `script is None`), it implies
                     that the current timepoint's script assignment is complete,
                     and the `script_received` event is set to signal the device thread.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set() # Signal to the DeviceThread that scripts are assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from the device's local store.

        Args:
            location (str): The identifier for the data location.

        Returns:
            Any: The sensor data at the specified location, or None if the location is not found.
        Functional Utility: Provides thread-safe read access to local sensor data.
        """
        with self.get_lock: # Acquire a lock to ensure exclusive read access.
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location in the device's local store.

        Args:
            location (str): The identifier for the data location.
            data (Any): The new sensor data value to be set.
        Functional Utility: Provides thread-safe write access to local sensor data.
        """
        with self.set_lock: # Acquire a lock to ensure exclusive write access.
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Waits for the DeviceThread to complete its execution before the program exits.
        Functional Utility: Ensures a graceful termination of the device's processing.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main processing thread for a Device. It is responsible for:
    1. Fetching scripts assigned to its Device.
    2. Dispatching these scripts to a pool of `ExecutorThread`s for parallel execution.
    3. Synchronizing with other DeviceThreads via a `ReusableBarrier`.
    """
    def __init__(self, device):
        """
        Initializes the DeviceThread with a reference to its parent Device.

        Args:
            device (Device): The Device instance this thread is managing.
        Functional Utility: Sets up the thread's context, linking it to its Device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.
        It orchestrates the processing of scripts across timepoints,
        managing worker threads and network-wide synchronization.
        """
        self.device.ready.wait() # Wait until shared resources (barrier, locks) are set up.
        while True:
            thread_list = [] # List to hold ExecutorThread instances for current timepoint.
            # Block Logic: Retrieves the list of neighboring devices from the supervisor.
            #              If `None` is returned, it signifies the end of the simulation.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit the main loop if no more neighbors, implying simulation end.

            # Block Logic: Waits for scripts to be assigned for the current timepoint.
            self.device.script_received.wait() 
            self.device.script_received.clear() # Clear the event for the next timepoint.

            pos = 0 # Tracks the position in the scripts list.

            # Block Logic: Creates an ExecutorThread for each assigned script.
            for _ in self.device.scripts:
                thread_list.append(ExecutorThread(self.device, self.device.scripts, neighbours, pos))
                pos = pos + 1

            scripts_left = len(self.device.scripts) # Total scripts to execute in this timepoint.
            current_pos = 0 # Starting index for the current batch of scripts.

            # Block Logic: Manages the parallel execution of `ExecutorThread`s.
            #              Scripts are processed in batches of up to 8 threads concurrently.
            # Functional Utility: Optimizes script execution by leveraging parallelism.
            if scripts_left < 8:
                # If fewer than 8 scripts, run all of them concurrently.
                for thread in thread_list:
                    thread.start()
                for thread in thread_list:
                    thread.join() # Wait for all threads in this batch to complete.
            else:
                while scripts_left >= 8:
                    # Execute scripts in batches of 8.
                    for i in xrange(current_pos, current_pos + 8):
                        thread_list[i].start()
                    for i in xrange(current_pos, current_pos + 8):
                        thread_list[i].join()
                    current_pos = current_pos + 8
                    scripts_left = scripts_left - 8

                # Execute any remaining scripts (fewer than 8).
                for i in xrange(current_pos, current_pos + scripts_left):
                    thread_list[i].start()
                for i in xrange(current_pos, current_pos + scripts_left):
                    thread_list[i].join()
            
            # After all scripts for the current timepoint are executed, synchronize.
            # Functional Utility: Ensures all devices complete their script processing
            #                     for the current timepoint before proceeding.
            self.device.reusable_barrier.wait()

class ExecutorThread(Thread):
    """
    A worker thread responsible for executing a single script for a Device.
    It fetches data from the local device and its neighbors, runs the script,
    and then propagates the results back to the local and neighboring devices.
    """
    def __init__(self, device, scripts, neighbours, pos):
        """
        Initializes an ExecutorThread.

        Args:
            device (Device): The parent Device object.
            scripts (list): The list of all scripts assigned to the parent Device.
            neighbours (list): A list of neighboring Device objects.
            pos (int): The index of the specific script within `scripts` that this
                       ExecutorThread is responsible for executing.
        Functional Utility: Sets up the context for executing a single script.
        """
        Thread.__init__(self, name="Executor Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours
        self.pos = pos

    def run(self):
        """
        Executes the assigned script.
        Functional Utility: Handles data acquisition, script execution, and result propagation
                            for a specific script and location.
        """
        (script, location) = self.scripts[self.pos] # Get the specific script and its location.
        
        # Block Logic: Acquire the location-specific lock to ensure exclusive access
        #              to data for this location across all devices during processing.
        # Pre-condition: `location` is a valid index into `self.device.location_lock`.
        with self.device.location_lock[location]: 
            script_data = [] # List to accumulate data required by the script.
            
            # Block Logic: Collects data from all neighboring devices for the current location.
            for dev in self.neighbours:
                data = dev.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Collects data from the local device for the current location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
            
            if script_data != []:
                # Functional Utility: Executes the script's `run` method with the collected data.
                result = script.run(script_data)
                
                # Block Logic: Propagates the result of the script execution to all neighboring devices.
                for dev in self.neighbours:
                    dev.set_data(location, result)
                
                # Updates the local device's sensor data with the script's result.
                self.device.set_data(location, result)
        # The location lock is automatically released upon exiting the 'with' block.
