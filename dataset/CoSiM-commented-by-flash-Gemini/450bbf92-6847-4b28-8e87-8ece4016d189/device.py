

"""
This module provides a complex, multi-threaded implementation for simulating a device
within a distributed system. It features a highly granular synchronization mechanism
to manage concurrent script execution and data access. Each logical `Device`
orchestrates up to eight internal `DeviceThread` instances, each acting as a worker.

Key Features:
- Utilizes `ReusableBarrierCond` (imported from `my_barrier.py`) for global synchronization
  across all `DeviceThread` instances in the system.
- Employs numerous `Event` and `Lock` objects for fine-grained control over script
  assignment, timepoint completion, and data access at various stages.
- `Device` class manages shared resources and initial setup.
- `DeviceThread` instances act as dedicated workers, fetching neighbor information,
  processing scripts, and participating in barrier synchronization.

This architecture aims to simulate complex distributed device interactions
with explicit concurrency control.
"""

from threading import Event, Thread, Lock
from my_barrier import ReusableBarrierCond

class Device(object):
    """
    Represents an individual simulated device in a distributed system.
    Each `Device` instance manages its own sensor data, communicates with a
    central supervisor, and orchestrates the execution of assigned scripts
    across multiple internal `DeviceThread` instances. It employs a complex
    synchronization scheme involving shared barriers, events, and locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping location IDs to their
                                current sensor data values.
            supervisor (object): A reference to the central supervisor managing
                                 the distributed system.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Lists of Event and Lock objects, one for each internal DeviceThread (8 threads).
        self.script_received = [] # Event for each thread to signal script assignment.
        self.scripts = []         # List of script lists, one for each thread.
        self.new_scripts = []     # List of new script lists, one for each thread.
        self.timepoint_done = []  # Event for each thread to signal timepoint completion.
        self.threads = []         # List to hold references to internal DeviceThread instances.
        self.nxt_thr_to_rcv_scr = 0 # Index to determine which DeviceThread receives the next script.
        
        self.data_access = Lock() # A lock for protecting this device's internal data (e.g., sensor_data).
        self.scripts_access = []  # List of Locks, one for each thread's script list.
        self.new_scripts_access = [] # List of Locks, one for each thread's new script list.
        
        # Shared synchronization primitives initialized during setup_devices.
        self.barrier1 = None      # First shared barrier.
        self.barrier2 = None      # Second shared barrier (declared but not used in current code).
        self.locs_acc = []        # List of Locks for location-specific data access.
        self.neighbours = None    # List of neighboring devices, populated by supervisor.


    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization primitives and spawns
        multiple `DeviceThread` instances for this device.
        The device with `device_id == 0` is responsible for creating global resources.

        Args:
            devices (list): A list of Device objects that are part of the same group.
        """
        # Block Logic: Device with ID 0 is responsible for creating the shared barrier.
        if self.device_id == 0:
            # Create a shared ReusableBarrierCond for all (num_devices * 8) DeviceThread instances.
            bar1 = ReusableBarrierCond(len(devices) * 8)
            # Distribute the barrier to all devices in the group.
            for dev in devices:
                dev.barrier1 = bar1
                
        # Block Logic: Spawn 8 internal DeviceThread instances for this Device.
        for i in range(8):
            self.threads.append(DeviceThread(self, i)) # Create DeviceThread with its ID.
            self.threads[i].start()                    # Start the DeviceThread.
            
            # Initialize corresponding synchronization primitives for each thread.
            self.scripts.append([])
            self.new_scripts.append([])
            self.script_received.append(Event())
            self.timepoint_done.append(Event())
            self.scripts_access.append(Lock())
            self.new_scripts_access.append(Lock())
        
        # Block Logic: Device with ID 0 initializes location-specific locks (`locs_acc`).
        if self.device_id == 0:
            max_loc = -1
            # Find the maximum location ID across all devices to determine size of `locs_acc`.
            for dev in devices:
                for loc in dev.sensor_data.keys():
                    if loc > max_loc:
                        max_loc = loc
            locs_locks = [] # Temporary list for location locks.
            # Create a Lock for each possible location ID.
            for i in range(max_loc+1):
                locs_locks.append(Lock())
            # Distribute the list of location locks to all devices.
            for dev in devices:
                dev.locs_acc = locs_locks



    def assign_script(self, script, location):
        """
        Assigns a script to one of the internal `DeviceThread` instances for execution.
        If `script` is `None`, it signals the end of script assignments for the timepoint.

        Args:
            script (object or None): The script object to execute, or `None`.
            location (object): The identifier for the location associated with the script.
        """
        # Block Logic: If a script is provided, assign it to the next available internal thread.
        if script is not None:
            i = self.nxt_thr_to_rcv_scr
            self.nxt_thr_to_rcv_scr = (self.nxt_thr_to_rcv_scr + 1) % 8 # Round-robin assignment.
            self.new_scripts[i].append((script, location)) # Add to the thread's new scripts list.
            self.script_received[i].set()                  # Signal to the thread that a script is received.
        # Block Logic: If `script` is None, signal all internal threads about timepoint completion.
        else:
            for j in range(8):
                self.timepoint_done[j].set()
                self.script_received[j].set()



    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's internal state.

        Args:
            location (object): The identifier of the location for which to retrieve data.

        Returns:
            Any: The sensor data associated with the location, or `None` if not found.
        """
        # Note: In this implementation, `data_access` is for `sensor_data` itself,
        # but location-specific access is managed by `locs_acc` in `DeviceThread`.
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location on this device.

        Args:
            location (object): The identifier of the location to update.
            data (Any): The new sensor data value for the location.
        """
        # Note: In this implementation, `data_access` is for `sensor_data` itself,
        # but location-specific access is managed by `locs_acc` in `DeviceThread`.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for all internal `DeviceThread` instances.
        This method waits for each internal thread to complete its execution.
        """
        for i in range(8):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread operating within a `Device` instance. Each `DeviceThread`
    is responsible for processing a subset of scripts assigned to its parent device.
    It participates in global synchronization with other `DeviceThread` instances
    across all devices using shared barriers, fetches neighbor information,
    and executes scripts while managing location-specific data access through locks.
    """
    

    def __init__(self, device, id_thread):
        """
        Initializes a DeviceThread worker.

        Args:
            device (Device): The parent Device instance this thread belongs to.
            id_thread (int): A unique identifier for this specific `DeviceThread` instance (0-7).
        """
        Thread.__init__(self, name="Device %d Thread %d" % (device.device_id, id_thread))
        self.device = device
        self.crt_tp = 0     # Current timepoint (used by thread 0).
        self.id_thread = id_thread # Unique ID for this thread (0-7).

    def run(self):
        """
        The main execution loop for this `DeviceThread`. It manages timepoint
        progression, fetches neighbors, processes both existing and newly assigned
        scripts, and participates in global barrier synchronization.
        """
        while True:
            # Block Logic: Only thread with id_thread == 0 is responsible for fetching neighbors
            # and updating the current timepoint for its parent device.
            if self.id_thread is 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.crt_tp += 1
            # Functional Utility: Global barrier synchronization. All DeviceThread instances
            # across all devices must wait here for `id_thread == 0` to fetch neighbors.
            self.device.barrier1.wait()
            
            # Pre-condition: If neighbors is None, it signals the end of simulation.
            if self.device.neighbours is None:
                break # Exit the thread's main loop.

            # Block Logic: Process existing scripts assigned to this specific DeviceThread.
            # These are scripts that were assigned in a previous timepoint but perhaps
            # waiting for some condition.
            for (script, location) in self.device.scripts[self.id_thread]:
                self.device.locs_acc[location].acquire() # Acquire lock for location-specific data.
                self.procces_script(script, location, self.device, self.device.neighbours)
                self.device.locs_acc[location].release() # Release lock for location-specific data.

            # Functional Utility: Signal that this thread has completed processing its scripts for this timepoint.
            self.device.timepoint_done[self.id_thread].wait()
            self.device.timepoint_done[self.id_thread].clear()

            # Block Logic: Process newly assigned scripts for this specific DeviceThread.
            # These are scripts assigned by the assign_script method in the current timepoint.
            for (script, location) in self.device.new_scripts[self.id_thread]:
                self.device.locs_acc[location].acquire() # Acquire lock for location-specific data.
                self.procces_script(script, location, self.device, self.device.neighbours)
                self.device.locs_acc[location].release() # Release lock for location-specific data.
                # Move newly processed script to the list of existing scripts for future timepoints.
                self.device.scripts[self.id_thread].append((script, location))

            # Functional Utility: Clear the new scripts list after processing.
            self.device.new_scripts[self.id_thread] = []
            
            # Functional Utility: Global barrier synchronization. All DeviceThread instances
            # must wait here after processing scripts before proceeding to the next timepoint.
            self.device.barrier1.wait()

    def procces_script(self, script_func, location, crt_device, neighbours):
        """
        Helper method to execute a single script for a given location.
        It collects relevant data, runs the script, and propagates the results.

        Args:
            script_func (object): The script function to execute.
            location (object): The identifier for the location.
            crt_device (Device): The current device instance.
            neighbours (list): A list of neighboring Device objects.
        """
        script_data = [] # List to accumulate data for the script.
        # Block Logic: Gather data from neighboring devices.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Gather data from the current device itself.
        data = crt_device.get_data(location)
        if data is not None:
            script_data.append(data)

        # Pre-condition: Check if any data was collected before running the script.
        if script_data != []:
            # Functional Utility: Execute the script with the collected data.
            result = script_func.run(script_data)

            # Block Logic: Propagate the result of the script execution back to all neighboring devices.
            for device in neighbours:
                device.set_data(location, result)
            
            # Block Logic: Update the current device's own sensor data with the result.
            self.device.set_data(location, result)
