# -*- coding: utf-8 -*-
"""
This module defines a simulation framework for a network of communicating devices.
It uses multi-threading to simulate concurrent device operations, with a focus on
data processing and synchronization across devices in discrete time steps.

Classes:
    Device: Represents a node in the network with its own sensor data and thread.
    RunScript: A thread to execute a data processing script on a device.
    DeviceThread: The main control loop for a device, handling synchronization.
"""

from threading import Event, Thread, Semaphore, Lock, RLock
from reusable_barrier import ReusableBarrier
import multiprocessing

class Device(object):
    """
    Represents a single device in a simulated distributed network.

    Each device runs its own thread and can be assigned scripts to process
    sensor data. It coordinates with other devices using synchronization
    primitives like barriers and locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device object.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor readings.
            supervisor (object): A central supervisor object that coordinates the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # --- Synchronization Primitives ---
        self.lock = None  # To be initialized in setup_devices
        self.dislocksdict = None  # Dictionary of location-based locks, initialized by device 0
        self.barrier = None  # Reusable barrier for timepoint synchronization
        self.sem = Semaphore(1)  # Used for initial setup signaling
        self.sem2 = Semaphore(0) # Used to ensure setup is complete before proceeding
        self.script_received = Event() # Not actively used, but part of the original design
        self.timepoint_done = Event()  # Signals that scripts for a timepoint are received

        # --- State ---
        self.scripts = []  # List of (script, location) tuples for the current timepoint
        
        # --- Main thread ---
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization objects for all devices in the simulation.

        The device with device_id 0 acts as the leader, creating the shared barrier
        and the dictionary of location-specific locks. It then distributes these
        to all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Collect all unique sensor locations from all devices.
        all_locations = set()
        for d in devices:
            all_locations.update(d.sensor_data.keys())

        # --- Leader-based Initialization (Device 0) ---
        # A single device (ID 0) is responsible for creating shared resources
        # to ensure all devices use the same instances.
        if self.device_id == 0:
            self.sem2.release()  # Allows this device to pass the first acquire()
            self.barrier = ReusableBarrier(len(devices))
            # Create a re-entrant lock for each unique sensor location.
            self.dislocksdict = {loc: RLock() for loc in all_locations}
            self.lock = Lock() # Own lock for data access

        # All devices wait here until the leader has finished initialization.
        self.sem2.acquire()

        # Distribute the shared objects to all other devices.
        for d in devices:
            if d.barrier is None:
                d.barrier = self.barrier
                d.dislocksdict = self.dislocksdict
                d.lock = Lock() # Each device gets its own lock for personal data
                d.sem2.release() # Signal to the next device that it can proceed

    def assign_script(self, script, location):
        """

        Assigns a script to be executed by the device for the current timepoint.

        Args:
            script (object): The script object with a `run` method.
            location (str): The location context for the script.
        
        If `script` is None, it signals that all scripts for the timepoint have been assigned.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # All scripts for this timepoint have been assigned; signal the device thread.
            self.timepoint_done.set()
   
    def get_data(self, location):
        """
        Thread-safely retrieves sensor data for a given location.

        Returns:
            The sensor data value, or None if the location is not relevant to this device.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Thread-safely sets sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class RunScript(Thread):
    """A thread responsible for executing a single script."""
    def __init__(self, script, location, neighbours, device):
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        Executes the script.

        The process involves:
        1. Acquiring a lock for the specific data location to ensure exclusive access.
        2. Gathering data from this device and all its neighbours for that location.
        3. Running the script on the collected data.
        4. Broadcasting the result by updating the data on this device and all neighbours.
        5. Releasing the location lock.
        """
        # Acquire a lock specific to the location, preventing concurrent modification
        # of this location's data by other scripts.
        self.device.dislocksdict[self.location].acquire()
        
        script_data = []
        # --- Data Gathering Phase ---
        # Pre-condition: Lock for `self.location` is held.
        # Collect data from all neighbouring devices.
        for device in self.neighbours:
            device.lock.acquire()
            data = device.get_data(self.location)
            device.lock.release()
            if data is not None:
                script_data.append(data)
                
        # Collect data from the current device.
        self.device.lock.acquire()
        data = self.device.get_data(self.location)
        self.device.lock.release()
        if data is not None:
            script_data.append(data)

        # --- Script Execution and Data Update ---
        if script_data:  # Only run if there is data to process.
            # Functional Utility: The script performs a computation (e.g., average, max)
            # on data from multiple sources.
            result = self.script.run(script_data)
            
            # --- Data Dissemination Phase ---
            # Update the data on all neighbours with the new result.
            for device in self.neighbours:
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()
            # Update the data on the current device.
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()
        
        # Post-condition: Data at `self.location` is consistent across this device and its neighbors.
        self.device.dislocksdict[self.location].release()


class DeviceThread(Thread):
    """The main control loop for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """

        The main simulation loop, advancing in discrete timepoints.
        
        The synchronization logic is as follows:
        1. Wait for the supervisor to signal that all scripts for a timepoint are assigned.
        2. **First Barrier**: All devices wait here. Ensures all devices have received their
           scripts before any device starts execution.
        3. Execute all assigned scripts in parallel threads.
        4. **Second Barrier**: All devices wait here. Ensures all devices have finished
           their computations for the timepoint before the next timepoint begins.
        """
        while True:
            # Get the list of neighbours for the current timepoint from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Supervisor signals shutdown.
                break

            # Block until the supervisor calls `assign_script(None, ...)`.
            self.device.timepoint_done.wait()

            # --- First Synchronization Point ---
            # Invariant: All devices have received their scripts for the current timepoint.
            self.device.barrier.wait()
            
            # Create and start threads for each script.
            script_threads = [
                RunScript(script, location, neighbours, self.device)
                for script, location in self.device.scripts
            ]
            for t in script_threads:
                t.start()
            
            # Wait for all script threads to complete their execution.
            for t in script_threads:
                t.join()
            
            # --- Second Synchronization Point ---
            # Invariant: All devices have finished processing for the current timepoint.
            self.device.barrier.wait()
            
            # Reset for the next timepoint.
            self.device.timepoint_done.clear()
            self.device.scripts = []