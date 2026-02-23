"""
This module provides a complex, multi-threaded framework for simulating a
distributed system of devices.

Each device operates with a pool of its own worker threads. The simulation appears
to be structured in discrete timepoints, synchronized globally with barriers. A key feature
is the use of fine-grained, location-based locking to manage concurrent access to shared
sensor data locations across all devices and threads.
"""

from threading import Event, Thread, Lock
from my_barrier import ReusableBarrierCond

class Device(object):
    """Represents a single, multi-threaded device in the distributed simulation.

    Each Device manages a pool of 8 internal DeviceThreads, distributing incoming
    scripts among them. It coordinates with other devices to set up shared barriers
    and a global set of locks for data locations.

    Attributes:
        device_id (int): A unique identifier for the device.
        threads (list): A list of 8 DeviceThread workers.
        locs_acc (list): A shared list of locks, one for each data location in the
                         entire system, to prevent race conditions.
        barrier1 (ReusableBarrierCond): A barrier to synchronize all threads from all
                                        devices at different stages of a timepoint.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance and its internal data structures."""
        self.device_id = device_id


        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = []
        self.scripts = []
        self.new_scripts = []
        self.timepoint_done = []
        self.threads = []
        # Used for round-robin assignment of scripts to threads.
        self.nxt_thr_to_rcv_scr = 0
        self.data_access = Lock() # Appears to be unused in favor of locs_acc.
        self.scripts_access = []
        self.new_scripts_access = []
        self.barrier1 = None
        self.barrier2 = None # Declared but not used.
        self.locs_acc = [] 
        self.neighbours = None


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources for all devices and starts their threads.

        Device 0 acts as a coordinator to create and distribute a shared barrier and
        a global set of location-based locks. This method also initializes and starts
        the 8 worker threads for the current device.
        """
        # Device 0 creates a barrier for all threads of all devices.
        if self.device_id == 0:
            bar1 = ReusableBarrierCond(len(devices) * 8)
            
            for dev in devices:
                dev.barrier1 = bar1
                
        # Initialize this device's 8 worker threads and their resources.
        for i in range(8):
            self.threads.append(DeviceThread(self, i))
            self.threads[i].start()
            self.scripts.append([])
            self.new_scripts.append([])
            self.script_received.append(Event())
            self.timepoint_done.append(Event())
            self.scripts_access.append(Lock()) # Appears unused.
            self.new_scripts_access.append(Lock()) # Appears unused.
        
        # Device 0 determines the max location ID to create a global lock for each location.
        if self.device_id == 0:
            max_loc = -1
            for dev in devices:
                for loc in dev.sensor_data.keys():
                    if loc > max_loc:
                        max_loc = loc
            locs_locks = [] 
            for i in range(max_loc+1):
                locs_locks.append(Lock())
            # Distribute the shared list of locks to all devices.
            for dev in devices:
                dev.locs_acc = locs_locks



    def assign_script(self, script, location):
        """Assigns a script to one of the device's worker threads in a round-robin fashion."""


        if script is not None:
            # Assign the script to the next thread in the rotation.
            i = self.nxt_thr_to_rcv_scr
            self.nxt_thr_to_rcv_scr = (self.nxt_thr_to_rcv_scr + 1) % 8
            self.new_scripts[i].append((script, location))
            # Signal the corresponding thread that a new script is available.
            self.script_received[i].set()
        else:
            # A 'None' script signals the end of a timepoint for all threads.
            for j in range(8):
                self.timepoint_done[j].set()
                self.script_received[j].set()



    def get_data(self, location):
        """Retrieves sensor data. Note: Not thread-safe on its own."""

        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data. Note: Not thread-safe on its own."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to complete."""
        for i in range(8):


            self.threads[i].join()


class DeviceThread(Thread):
    """A worker thread for a Device. Each device runs 8 of these."""

    def __init__(self, device, id_thread):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device %d Thread %d" % (device.device_id, id_thread))
        self.device = device
        self.crt_tp = 0 # Timepoint counter, seems only used by thread 0.
        self.id_thread = id_thread

    def run(self):
        """The main execution loop for the worker thread.
        
        The loop is synchronized by a global barrier. It processes old scripts,
        waits for new scripts for the current timepoint, processes them, and then
        synchronizes again. It uses location-specific locks to ensure data consistency.
        """

        while True:
            
            # Block Logic: Start of a timepoint, synchronized across all threads of all devices.
            # Thread 0 of each device has the special role of fetching neighbor info.
            if self.id_thread is 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.crt_tp += 1
            self.device.barrier1.wait()
            
            if self.device.neighbours is None:
                # Supervisor signals simulation end by returning None for neighbours.
                break

            # Process scripts from the previous timepoint.
            for (script, location) in self.device.scripts[self.id_thread]:
                self.device.locs_acc[location].acquire()
                self.procces_script(script, location, self.device, self.device.neighbours)
                self.device.locs_acc[location].release()

            # Wait until the supervisor signals that the current timepoint is done.
            self.device.timepoint_done[self.id_thread].wait()
            self.device.timepoint_done[self.id_thread].clear()

            # Process newly assigned scripts for this timepoint.
            for (script, location) in self.device.new_scripts[self.id_thread]:
                self.device.locs_acc[location].acquire()
                self.procces_script(script, location, self.device, self.device.neighbours)
                self.device.locs_acc[location].release()
                # Move the new script to the old scripts list for the next cycle.
                self.device.scripts[self.id_thread].append((script, location))

            self.device.new_scripts[self.id_thread] = []
            
            # Block Logic: End of a timepoint. Synchronize before the next begins.
            self.device.barrier1.wait()

    def procces_script(self, script_func, location, crt_device, neighbours):
        """Gathers data, runs a script, and distributes the result.
        
        Note: This method assumes the appropriate location-based lock has already been acquired.
        """
        script_data = []
        # Gather data from neighbors.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the current device.
        data = crt_device.get_data(location)
        if data is not None:
            script_data.append(data)
        if script_data != []:
            # Execute the script function with the collected data.
            result = script_func.run(script_data)

            # Propagate the result back to all neighbors and the current device.
            for device in neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)
