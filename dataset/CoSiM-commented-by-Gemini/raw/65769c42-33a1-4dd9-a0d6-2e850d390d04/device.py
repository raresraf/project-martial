"""
@file device.py
@brief Defines a device model for a simulation with a multi-level locking scheme.

This file implements a `Device` that uses a dedicated `ScriptThread` for each
script execution. Synchronization is managed through a shared barrier and
several shared locks, including a global script lock and a map of location-
specific locks, all initialized by a single root device.
"""

from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier


class Device(object):
    """
    Represents a device in the simulation, managing a hierarchy of shared locks.

    Synchronization objects are stored as class-level attributes and are shared
    across all device instances after being initialized by the root device.
    """
    
    # Class-level attributes for shared synchronization primitives.
    timepoint_barrier = None
    script_lock = None # A global lock for writing results.
    data_lock = None   # An apparently unused global lock.
    data_locks = {}    # A map of location-specific locks.

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.devices = []

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.

        Executed by the root device (ID 0), this method creates and assigns the
        shared barrier and global locks to all devices in the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices
        num_devices = len(devices)
        if self.device_id == 0:
            self.timepoint_barrier = ReusableBarrier(num_devices)
            self.script_lock = Lock()
            self.data_lock = Lock() # This lock is initialized but never used.
            # Invariant: Propagate the shared objects to all other devices.
            for i in range(1, len(devices)):
                devices[i].data_lock = self.data_lock
                devices[i].script_lock = self.script_lock
                devices[i].timepoint_barrier = self.timepoint_barrier

    def assign_script(self, script, location):
        """
        Assigns a script and lazily initializes a location-specific lock if needed.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
            # Block Logic: If a lock for this location doesn't exist, create it
            # and distribute it to all other devices.
            if not location in self.data_locks:
                lock = Lock()
                for dev in self.devices:
                    dev.data_locks[location] = lock
        else:
            # A None script signals that script assignment is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. This access is not synchronized by this method."""
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        """Updates sensor data. This access is not synchronized by this method."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, which spawns worker threads for scripts.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script_threads = []

    def run(self):
        """The main simulation loop."""
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Create and start a worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                script_thread = ScriptThread(self.device, script, location, \
                    neighbours)
                script_thread.start()
                self.script_threads.append(script_thread)

            
            self.device.timepoint_done.clear()
            # Wait for all script threads to complete their execution.
            for script_thread in self.script_threads:
                script_thread.join()
            self.script_threads = []
            
            # Wait at the barrier for all other devices to finish their timepoint.
            self.device.timepoint_barrier.wait()

class ScriptThread(Thread):
    """
    A worker thread responsible for executing a single script with two-level locking.
    """

    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script using a location-specific lock and a global script lock.
        """
        # Pre-condition: Acquire the lock specific to this script's location.
        with self.device.data_locks[self.location]:
            script_data = []
            
            # Block Logic: Gather data from neighbors and the local device.
            # This is safe because of the location-specific lock.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = self.script.run(script_data)

                # Pre-condition: Acquire a global lock before writing back the result.
                # This serializes the write-back phase across all scripts for all
                # locations, which may be overly restrictive.
                with self.device.script_lock:
                    
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    
                    self.device.set_data(self.location, result)