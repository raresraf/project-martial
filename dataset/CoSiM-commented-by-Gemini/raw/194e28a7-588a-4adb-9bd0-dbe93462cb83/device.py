"""
This module defines a device simulation framework using a "thread-per-script"
execution model.

The architecture consists of four main classes:
- ReusableBarrierCond: A barrier for synchronizing multiple threads.
- ScriptThread: A dedicated thread created to execute a single script.
- Device: Represents a node in the network, managed by a master DeviceThread.
- DeviceThread: The master thread for a device, which dynamically spawns
  a new ScriptThread for every assigned script in each time step.
"""
from threading import Condition, Thread, Event, RLock


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


class ScriptThread(Thread):
    """A worker thread that is created to execute exactly one script."""
    def __init__(self, script, device, location, neighbours):
        """
        Initializes the ScriptThread.

        Args:
            script (Script): The script object to execute.
            device (Device): The parent device.
            location (any): The data location for the script.
            neighbours (list): A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.script = script
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.locks = device.locks # Reference to the globally shared locks.

    def run(self):
        """
        Executes the script, ensuring serialized access to the data location.
        """
        script_data = []

        # Acquire the global lock for this specific data location. This ensures
        # that only one thread in the entire simulation can operate on this
        # location at any given time.
        with self.locks.get(self.location):
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            # Gather data from self.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Execute the script and propagate the results.
            if script_data:
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class Device(object):
    """
    Represents a single device in the simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        
        # --- Globally Shared Objects (set by creator) ---
        self.barrier = None
        self.locks = None
        
        # Flag to identify the single device responsible for setup.
        self.creator = True

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global synchronization objects for all devices.

        The first device to call this method (the "creator") is responsible for
        instantiating and distributing the shared barrier and locks dictionary.
        """
        if self.creator is True:
            barrier = ReusableBarrierCond(len(devices))
            locks = {}
            # Initialize locks for the creator's own locations.
            for location in self.sensor_data:
                if not locks.get(location):
                    locks[location] = RLock()
            # Distribute shared objects to all devices.
            for device in devices:
                device.barrier = barrier
                device.locks = locks
                device.creator = False

    def assign_script(self, script, location):
        """
        Assigns a script to be executed in the current time step.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of script assignment for this step.
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

    The main control thread for a Device. It uses a "thread-per-script" model.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main lifecycle loop of the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # End of simulation.

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            script_threads = []
            # --- Thread-per-Script Creation ---
            # For every script assigned in this time step, create a new thread.
            for (script, location) in self.device.scripts:
                # Lazily create a lock for a location if not seen before.
                if not self.device.locks.get(location):
                    self.device.locks[location] = RLock()
                
                script_thread = ScriptThread(script, self.device, location, neighbours)
                script_threads.append(script_thread)

            # Start all worker threads for this time step.
            for script_thread in script_threads:
                script_thread.start()

            # Wait for all worker threads to complete.
            for script_thread in script_threads:
                script_thread.join()

            # --- Global Synchronization ---
            # Wait at the barrier for all other devices to finish their time step.
            self.device.barrier.wait()
