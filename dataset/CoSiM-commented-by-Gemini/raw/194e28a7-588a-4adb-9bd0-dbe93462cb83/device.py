"""
This module defines a distributed device simulation framework where each device
has a controller thread that dynamically spawns a new worker thread for each
assigned script in every simulation step.
"""

from threading import Condition
from threading import Thread, Event, RLock


class ReusableBarrierCond(object):
    """
    A custom implementation of a reusable barrier using a Condition variable.
    
    This barrier blocks a set of threads until all of them have called the
    `wait()` method, at which point it releases them all and resets for reuse.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a specific number of threads."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class ScriptThread(Thread):
    """A short-lived worker thread designed to execute a single script."""

    def __init__(self, script, device, location, neighbours):
        """
        Initializes the worker thread with a specific task.

        Args:
            script (Script): The script object to execute.
            device (Device): The parent device.
            location (any): The data location this script operates on.
            neighbours (list): A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.script = script
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.locks = device.locks # Reference to the shared dictionary of locks.

    def run(self):
        """
        Executes the script.

        This method acquires a global lock for the script's data location,
        ensuring that only one thread in the entire simulation can operate on
        this location at a time.
        """
        script_data = []
        
        # Acquire the global lock for this specific location.
        self.locks.get(self.location).acquire()

        # Gather data from neighbors and the local device.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # If data was found, run the script and write back the results.
        if not script_data == []:
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

        # Release the global lock for the location.
        self.locks.get(self.location).release()


class Device(object):
    """
    Represents a device node which is managed by a controller thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and its main controller thread.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The local data for this device.
            supervisor (Supervisor): The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # Signals the controller to start a step.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None # The shared barrier for all devices.
        self.locks = None   # The shared dictionary of location locks.
        self.creator = True # Flag to identify the master device (for setup).

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.
        
        The device with `creator=True` acts as the master, creating a barrier
        and a dictionary of re-entrant locks for each data location.
        """
        if self.creator is True:
            barrier = ReusableBarrierCond(len(devices))
            locks = {}
            
            # Create a lock for each known data location.
            for location in self.sensor_data:
                if not locks.get(location):
                    locks[location] = RLock()
            
            # Distribute the shared objects to all devices.
            for index in xrange(len(devices)):
                devices[index].barrier = barrier
                devices[index].locks = locks
                devices[index].creator = False
        else:
            pass

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current time step.
        
        A `None` script signals that all assignments are complete for the step.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data from the device's local sensor data.
        Note: This method is not thread-safe; locking is handled by the caller.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Sets data in the device's local sensor data.
        Note: This method is not thread-safe; locking is handled by the caller.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's controller thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main controller thread that orchestrates work for a single device."""

    def __init__(self, device):
        """Initializes the controller thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            # Get the list of neighbors for this time step from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the supervisor to signal that all scripts for this step are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # --- Dynamic Worker Thread Creation ---
            script_threads = []
            # For each script, create a new worker thread to execute it.
            for (script, location) in self.device.scripts:
                # Ensure a lock exists for the location.
                if not self.device.locks.get(location):
                    self.device.locks[location] = RLock()
                
                script_thread = \
                    ScriptThread(script, self.device, location, neighbours)
                script_threads.append(script_thread)

            # Start all worker threads for this step.
            for script_thread in script_threads:
                script_thread.start()

            # Wait for all worker threads to complete their tasks.
            for script_thread in script_threads:
                script_thread.join()

            # Synchronize with all other devices before proceeding to the next step.
            self.device.barrier.wait()
