"""
This module provides another implementation of a distributed device simulation.

This version features a different concurrency model compared to others. Key aspects
include:
- A shared barrier object for all devices, initialized collectively.
- A shared dictionary of locks for data locations, also initialized collectively.
- A `DeviceThread` that dynamically creates and destroys a new set of worker
  threads (`ScriptThread`) at each simulation time step, which is a significant
  performance anti-pattern.
- Worker threads use a work-stealing model from a shared list to process scripts.
- Locking is handled by the worker threads, which acquire a global lock for a
  specific data location before accessing that data on any device.

Note: This script depends on a local `barrier.py` file and uses Python 2 syntax.
"""

from threading import Event, Thread, Lock
# Assumes a local file `barrier.py` with a ReusableBarrierSem implementation.
from barrier import ReusableBarrierSem as Barrier

class Device(object):
    """
    Represents a device node in the simulation.

    This class holds device-specific data and scripts. It participates in a
    collective setup of shared synchronization objects (barrier and locks)
    that are used by all devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device object and starts its control thread.

        :param device_id: The unique ID for this device.
        :param sensor_data: A dictionary of initial sensor data for this device.
        :param supervisor: The central supervisor object.
        """
        # These objects are intended to be shared across all device instances.
        # They are initialized to None and set up in `setup_devices`.
        self.timepoint_done = None  # The global, shared barrier.
        self.lock = None            # The global, shared dictionary of locks.
        self.todo_scripts = []      # A temporary list for work-stealing.

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs collective initialization of shared synchronization objects.

        This method is designed to be called by all devices. It ensures that a single
        shared barrier and a single shared lock dictionary are created and
        distributed to all device instances in the simulation.

        :param devices: A list of all Device objects.
        """
        # The first device to enter this block creates the shared objects.
        if self.timepoint_done is None:
            self.timepoint_done = Barrier(len(devices))
            for device in devices:
                if device.timepoint_done is None:
                    device.timepoint_done = self.timepoint_done

        if self.lock is None:
            self.lock = {}
            for device in devices:
                if device.lock is None:
                    device.lock = self.lock

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current timepoint and sets up locks.

        :param script: The script object to be executed. If None, it signals the
                       end of script assignment for the current timepoint.
        :param location: The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Dynamically create a shared lock for a location if it's the first
            # time we've seen it. This lock is shared by all devices.
            if location not in self.lock:
                self.lock[location] = Lock()
        else:
            # Signal that all scripts for this timepoint have been received.
            self.script_received.set()

    def get_data(self, location):
        """
        Gets data for a location. This method is NOT thread-safe by itself.
        Locking must be handled by the caller.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets data for a location. This method is NOT thread-safe by itself.
        Locking must be handled by the caller.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    It orchestrates the simulation's time steps by dynamically spawning and
    joining worker threads (`ScriptThread`) for each step.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop, synchronized by timepoints."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # Supervisor signals termination.

            # Wait until all scripts for the current timepoint have been assigned.
            self.device.script_received.wait()

            # Copy scripts to a temporary list for the worker threads to process.
            for script in self.device.scripts:
                self.device.todo_scripts.append(script)

            # Inefficiently create a new pool of worker threads for each timepoint.
            nr_subthreads = min(8, len(self.device.scripts))
            subthreads = []
            scripts_lock = Lock()

            while len(subthreads) < nr_subthreads:
                subthread = ScriptThread(scripts_lock, self, neighbours)
                subthreads.append(subthread)

            for subthread in subthreads:
                subthread.start()

            # Wait for all worker threads for this timepoint to complete.
            for subthread in subthreads:
                subthread.join()

            # Reset for the next timepoint.
            self.device.script_received.clear()

            # Wait at the global barrier for all other devices to finish this timepoint.
            self.device.timepoint_done.wait()

class ScriptThread(Thread):
    """
    A short-lived worker thread that executes scripts for a device.
    It uses a work-stealing approach to get scripts from a shared list.
    """

    def __init__(self, scripts_lock, parent, neighbours):
        Thread.__init__(self)
        self.scripts_lock = scripts_lock  # Lock for the shared 'todo_scripts' list.
        self.parent = parent
        self.neighbours = neighbours

    def run(self):
        """
        Continuously fetches and executes scripts until none are left.
        """
        # --- Work-stealing loop ---
        self.scripts_lock.acquire()
        length = len(self.parent.device.todo_scripts)
        if length > 0:
            current_script = self.parent.device.todo_scripts.pop()
        else:
            current_script = None
        self.scripts_lock.release()

        while current_script is not None and self.neighbours is not None:
            (script, location) = current_script
            script_data = []

            # Acquire the global lock for this specific data location before
            # accessing it on any device.
            self.parent.device.lock[location].acquire()

            # Gather data from all neighbors and the parent device itself.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = self.parent.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Run the script on the collected data.
                result = script.run(script_data)

                # Broadcast the result back to all neighbors and the parent device.
                for device in self.neighbours:
                    device.set_data(location, result)
                self.parent.device.set_data(location, result)

            self.parent.device.lock[location].release()

            # --- Attempt to steal the next piece of work ---
            self.scripts_lock.acquire()
            length = len(self.parent.device.todo_scripts)
            if length > 0:
                current_script = self.parent.device.todo_scripts.pop()
            else:
                current_script = None
            self.scripts_lock.release()
