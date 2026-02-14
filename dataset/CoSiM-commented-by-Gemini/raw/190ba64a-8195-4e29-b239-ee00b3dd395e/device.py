"""
This module defines a simulation framework for distributed devices.

The architecture is composed of several classes that model the devices and their
concurrent execution through a combination of dynamic thread creation and
global synchronization primitives.

Key components are:
- Device: Represents a node, controlled by a master `DeviceThread`.
- DeviceThread: The master thread for a device, which orchestrates time steps
  and dynamically spawns worker threads.
- MyThread: A worker thread that executes a subset of a device's scripts.
- ReusableBarrierCond: A helper class for barrier synchronization.
"""
from threading import Event, Thread, Lock, Condition


class ReusableBarrierCond(object):
    """
    A reusable barrier implementation using a Condition variable.

    Threads block on `wait()` until the required number of threads have
    arrived. The barrier then releases all threads and resets for future use.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Causes the calling thread to wait at the barrier."""
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                self.cond.notify_all()
                self.count_threads = self.num_threads
            else:
                self.cond.wait()


class Device(object):
    """
    Represents a single device in the simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's local data.
            supervisor (Supervisor): The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        
        # --- Synchronization Objects ---
        # Event to signal when all scripts for a time step are assigned.
        self.timepoint_done = Event()
        # A global barrier for time step synchronization (set by root device).
        self.barrier = None
        # A global dictionary of locks for data locations (set by root device).
        self.sync_location_lock = {}
        # A lock for this device's own data (appears to be a source of deadlocks).
        self.sync_data_lock = Lock()
        
        self.cores = 8
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global synchronization objects for all devices.

        This method must be called on the root device (device_id == 0).
        """
        if self.device_id == 0:
            # Discover all unique locations to create a lock for each.
            all_locations = set()
            for device in devices:
                for location in device.sensor_data:
                    all_locations.add(location)
            
            # Create shared lock and barrier objects.
            sync_location_lock = {loc: Lock() for loc in all_locations}
            barrier = ReusableBarrierCond(len(devices))
            
            # Distribute shared objects to all devices.
            for device in devices:
                device.barrier = barrier
                device.sync_location_lock = sync_location_lock

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current time step.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script from the supervisor signals the end of assignment.
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data from a specific location (not thread-safe by itself)."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a specific location (not thread-safe by itself)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating the time steps and
    dynamically creating worker threads (`MyThread`).
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main lifecycle loop of the device."""
        while True:
            # Wait until the supervisor signals that all scripts are assigned.
            self.device.timepoint_done.wait()

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.
            
            # --- Dynamic Thread Creation and Execution ---
            # A new set of worker threads is created for every time step.
            my_threads = [MyThread(self) for _ in range(self.device.cores)]
            
            # Distribute scripts to workers in a round-robin fashion.
            index = 0
            for (script, location) in self.device.scripts:
                my_threads[index % self.device.cores].assign_script(script, location)
                index += 1
            
            # Start and join all worker threads for this time step.
            for thread in my_threads:
                thread.set_neighbours(neighbours)
                thread.start()
            for thread in my_threads:
                thread.join()
            
            self.device.timepoint_done.clear()
            
            # --- Global Synchronization ---
            # Wait at the global barrier to ensure all devices have finished this step.
            self.device.barrier.wait()


class MyThread(Thread):
    """
    A worker thread that executes a batch of assigned scripts.
    """
    def __init__(self, parent_device_thread):
        Thread.__init__(self)
        self.parent = parent_device_thread
        self.scripts = []
        self.neighbours = []

    def set_neighbours(self, neighbours):
        """Sets the neighbor list for the current time step."""
        self.neighbours = neighbours

    def assign_script(self, script, location):
        """Adds a script to this thread's workload."""
        self.scripts.append((script, location))

    def run(self):
        """
        Executes the assigned scripts, handling data synchronization.
        """
        for (script, location) in self.scripts:
            # Acquire the global lock for this specific location.
            # This ensures only one thread in the entire simulation can work
            # on this location at a time.
            with self.parent.device.sync_location_lock[location]:
                script_data = []
                
                # The locking pattern below is problematic and can lead to deadlocks.
                # A single global lock per device (`sync_data_lock`) is acquired
                # repeatedly. The outer `sync_location_lock` should be sufficient.
                
                # Gather data from neighbors.
                for device in self.neighbours:
                    with device.sync_data_lock:
                        data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from self.
                with self.parent.device.sync_data_lock:
                    data = self.parent.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Run script and write results back.
                if script_data:
                    result = script.run(script_data)
                    for device in self.neighbours:
                        with device.sync_data_lock:
                            device.set_data(location, result)
                    with self.parent.device.sync_data_lock:
                        self.parent.device.set_data(location, result)
