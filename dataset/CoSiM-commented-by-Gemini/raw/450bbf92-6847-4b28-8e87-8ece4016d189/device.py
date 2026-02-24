"""
This module implements a complex, multi-threaded simulation framework for a
network of devices processing sensor data. It represents a more advanced design
compared to simpler device simulators.

Key architectural features include:
- Intra-device parallelism: Each `Device` instance spawns a fixed number of
  worker threads (`DeviceThread`), in this case, 8.
- Global, location-based locking: A global set of locks, one for each data
  location, is shared among all devices to ensure exclusive access to data
  points during script processing.
- Two-phase timepoints: The simulation appears to proceed in synchronized
  timepoints, but within each timepoint, threads first process scripts from the
  previous cycle and then process newly assigned scripts.
- Centralized setup: The device with ID 0 is responsible for initializing
  shared resources like the main barrier and the location-based locks.
"""

from threading import Event, Thread, Lock
# This appears to be a custom barrier implementation.
from my_barrier import ReusableBarrierCond

class Device(object):
    """Represents a single, multi-threaded device in the simulated network.

    Each device manages a pool of 8 worker threads, handles script assignment
    in a round-robin fashion, and coordinates with other devices using shared
    barriers and a global lock system for data locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor data values.
            supervisor (object): The supervisor object managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Per-thread state management
        self.script_received = []  # Events to signal new script arrival for a thread
        self.scripts = []          # List of scripts for each thread from previous timepoints
        self.new_scripts = []      # List of newly assigned scripts for each thread
        self.timepoint_done = []   # Events to signal the end of a processing phase
        self.threads = []
        self.scripts_access = []   # Per-thread locks for script lists
        self.new_scripts_access = []

        self.nxt_thr_to_rcv_scr = 0  # Round-robin counter for script assignment

        # Shared resources
        self.barrier1 = None         # Main synchronization barrier
        self.locs_acc = []           # Global locks for data locations
        self.neighbours = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources for all devices.

        If this is device 0, it creates a shared barrier for all threads of all
        devices. It also creates a global array of locks, one for each unique
        data location across all devices.

        This method also starts the 8 worker threads for the current device.

        Args:
            devices (list): A list of all Device objects in the network.
        """
        if self.device_id == 0:
            # Create a barrier for all threads across all devices (len(devices) * 8).
            bar1 = ReusableBarrierCond(len(devices) * 8)
            for dev in devices:
                dev.barrier1 = bar1

        # Initialize this device's 8 worker threads and their state containers.
        for i in range(8):
            self.threads.append(DeviceThread(self, i))
            self.threads[i].start()
            self.scripts.append([])
            self.new_scripts.append([])
            self.script_received.append(Event())
            self.timepoint_done.append(Event())
            self.scripts_access.append(Lock())
            self.new_scripts_access.append(Lock())
        
        # Device 0 sets up global location locks for all devices.
        if self.device_id == 0:
            max_loc = -1
            for dev in devices:
                for loc in dev.sensor_data.keys():
                    if loc > max_loc:
                        max_loc = loc
            
            locs_locks = [Lock() for _ in range(max_loc + 1)]
            for dev in devices:
                dev.locs_acc = locs_locks

    def assign_script(self, script, location):
        """Assigns a script to one of the device's threads.

        Scripts are distributed to the 8 worker threads in a round-robin fashion.
        A `None` script is a broadcast signal to all threads that the timepoint
        is ending.

        Args:
            script (object): The script to execute, or None to signal timepoint end.
            location (str): The data location for the script.
        """
        if script is not None:
            # Assign script to the next thread in the rotation.
            i = self.nxt_thr_to_rcv_scr
            self.nxt_thr_to_rcv_scr = (self.nxt_thr_to_rcv_scr + 1) % 8
            self.new_scripts[i].append((script, location))
            self.script_received[i].set()  # Signal the thread about the new script.
        else:
            # Broadcast the timepoint end signal to all threads.
            for j in range(8):
                self.timepoint_done[j].set()
                self.script_received[j].set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location.

        Note: This method is not thread-safe by itself. The caller (DeviceThread)
        is responsible for acquiring the appropriate location lock (`locs_acc`).
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data at a specific location.

        Note: This method is not thread-safe by itself. The caller (DeviceThread)
        is responsible for acquiring the appropriate location lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all worker threads to shut down the device."""
        for i in range(8):
            self.threads[i].join()


class DeviceThread(Thread):
    """A single worker thread within a Device.

    Each device has 8 of these. They perform the core work of processing scripts
    and synchronizing with other threads.
    """

    def __init__(self, device, id_thread):
        """Initializes the device thread.

        Args:
            device (Device): The parent device this thread belongs to.
            id_thread (int): The ID of this thread within the device (0-7).
        """
        Thread.__init__(self, name="Device %d Thread %d" % (device.device_id, id_thread))
        self.device = device
        self.crt_tp = 0
        self.id_thread = id_thread

    def run(self):
        """The main simulation loop for the worker thread."""
        while True:
            # --- Phase 1: Synchronization and Neighbor Discovery ---
            if self.id_thread == 0:
                # Thread 0 is responsible for fetching the list of neighbors for this device.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.crt_tp += 1
            
            # All threads of all devices synchronize here before starting the timepoint.
            self.device.barrier1.wait()
            
            if self.device.neighbours is None:
                # A None neighbor list is the signal to terminate.
                break

            # --- Phase 2: Process scripts from the previous timepoint ---
            for (script, location) in self.device.scripts[self.id_thread]:
                self.device.locs_acc[location].acquire()
                self.procces_script(script, location, self.device, self.device.neighbours)
                self.device.locs_acc[location].release()

            # Wait for the signal that new scripts for this timepoint have been assigned.
            self.device.timepoint_done[self.id_thread].wait()
            self.device.timepoint_done[self.id_thread].clear()

            # --- Phase 3: Process newly assigned scripts ---
            for (script, location) in self.device.new_scripts[self.id_thread]:
                self.device.locs_acc[location].acquire()
                self.procces_script(script, location, self.device, self.device.neighbours)
                self.device.locs_acc[location].release()
                # Add the new script to the list for the next timepoint's processing.
                self.device.scripts[self.id_thread].append((script, location))
            
            self.device.new_scripts[self.id_thread] = []

            # --- Phase 4: Final synchronization for the timepoint ---
            self.device.barrier1.wait()

    def procces_script(self, script_func, location, crt_device, neighbours):
        """Gathers data, runs a script, and distributes the result."""
        script_data = []
        # Gather data from all neighbors (assuming the list includes the current device).
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        if script_data:
            result = script_func.run(script_data)

            # Update data on all neighbors (and self).
            for device in neighbours:
                device.set_data(location, result)
