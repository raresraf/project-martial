"""
@file device.py
@brief Defines a device model for a distributed simulation with a custom barrier.

This file contains a `Device` class that uses a dedicated `ScriptThread` for
each script execution. It features a complex, custom-built barrier mechanism
for timepoint synchronization, where each device signals all other devices and
then waits multiple times on a semaphore.
"""

from threading import Event, Thread, Semaphore, current_thread, RLock


class Device(object):
    """
    Represents a device in a simulation, managing script execution and a
    complex, manual synchronization scheme.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): Unique ID for the device.
            sensor_data (dict): Dictionary of the device's sensor readings.
            supervisor: The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Semaphore(0) # Acts as a counter for assigned scripts.
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.all_devices = None
        self.timepoint_sem = Semaphore(0) # Used for the custom barrier.
        self.list_thread = []
        self.loc_lock = {} # Dictionary to hold re-entrant locks for locations.

    def __str__(self):
        """Returns a string representation of the device, including the current thread."""
        return "[%.35s]    Device %d:" % (current_thread(),self.device_id)

    def sync_on_timepoint(self):
        """
        Implements a waiting phase of a custom barrier.

        This method blocks by acquiring its own semaphore `N-1` times, where N is
        the total number of devices. It expects other devices to release its
        semaphore to unblock.
        """
        for i in range(len(self.all_devices)-1):
            self.timepoint_sem.acquire()

    def setup_devices(self, devices):
        """
        Records the list of all devices for later use in synchronization.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.all_devices = devices

    def assign_script(self, script, location):
        """
        Assigns a script to be executed and creates a lock for its location.

        Args:
            script: The script object to execute.
            location: The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.loc_lock[location] = RLock()
        else:
            # A None script signals the end of script assignment for the timepoint.
            self.timepoint_done.set()
        # Release the semaphore to signal that a script has been received.
        self.script_received.release()

    def lock(self, location):
        """Acquires the re-entrant lock for a specific location."""
        if not location in self.loc_lock:
            self.loc_lock[location] = RLock()
        self.loc_lock[location].acquire()

    def unlock(self, location):
        """
        Releases the re-entrant lock for a specific location.

        Note: The empty except block is generally unsafe as it can hide
        errors, such as releasing a lock that isn't held.
        """
        try:
            self.loc_lock[location].release()
        except:
            pass

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class ScriptThread(Thread):
    """
    A worker thread responsible for executing a single script and handling
    the associated complex cross-device locking.
    """
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script, managing locking across the local device and its neighbors.
        """
        script_data = []
        
        # Block Logic: Attempt to acquire locks on all neighbors for the given location
        # before gathering data. This is a complex and potentially deadlock-prone pattern.
        if self.neighbours:
            for device in self.neighbours:
                device.lock(self.location)
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
        
        # Gather data from the local device as well.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Invariant: Data is gathered and ready for script execution.
        if script_data != []:
            
            result = self.script.run(script_data)
            
            # Block Logic: Propagate the result back to all devices.
            if self.neighbours:
                for device in self.neighbours:
                    device.set_data(self.location, result)

            self.device.lock(self.location)
            self.device.set_data(self.location, result)
            self.device.unlock(self.location)

        # Block Logic: Release the locks on all neighboring devices.
        if self.neighbours:
            for device in self.neighbours:
                device.unlock(self.location)

class DeviceThread(Thread):
    """
    The main control thread for a Device, orchestrating script execution and
    the custom barrier synchronization.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Waits for the supervisor to signal that script assignment is complete.
            self.device.timepoint_done.wait()
            # Waits until at least one script has been assigned.
            self.device.script_received.acquire()

            # Block Logic: Spawn and start a worker thread (`ScriptThread`) for each script.
            for (script, location) in self.device.scripts:
                thread = ScriptThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)
                thread.start()

            # Wait for all script threads to complete.
            for t in self.device.list_thread:
                t.join()
            self.device.list_thread = []

            # --- Custom Barrier Implementation ---
            # Block Logic: Signaling phase. Each device releases the semaphore
            # of every other device in the simulation.
            for d in self.device.all_devices:
                if d == self.device:
                    continue
                d.timepoint_sem.release()

            # Reset the timepoint event for the next cycle.
            self.device.timepoint_done.clear()
            # Block Logic: Waiting phase. Each device waits to be signaled N-1 times.
            self.device.sync_on_timepoint()