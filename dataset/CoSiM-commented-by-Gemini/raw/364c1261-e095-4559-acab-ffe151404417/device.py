"""
A device simulation framework using a thread-per-script model and mixed locking.

This module implements a device simulation where a master device (ID 0) is
responsible for initializing shared resources like a barrier and global locks.
These resources are stored as class-level variables, shared across all device
instances. The system spawns a new thread for each script in a timepoint and
uses a combination of location-specific locks for reads and a single global
lock for all writes. The dynamic creation of location locks is not thread-safe.
"""

from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier


class Device(object):
    """
    Represents a device that participates in a master-led setup and uses a
    mix of shared class-level and instance-level resources for synchronization.
    """
    # Class-level variables shared by all instances of Device.
    timepoint_barrier = None
    script_lock = None
    data_lock = None
    data_locks = {}

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's internal sensor data.
            supervisor: The central supervisor managing the device network.
        """
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
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources using a master-device pattern.
        
        The device with ID 0 creates a shared barrier and global locks,
        storing them in class-level variables, making them accessible to all
        other device instances.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices
        num_devices = len(devices)
        if self.device_id == 0:
            self.timepoint_barrier = ReusableBarrier(num_devices)
            self.script_lock = Lock()
            self.data_lock = Lock()
            for i in range(1, len(devices)):
                devices[i].data_lock = self.data_lock
                devices[i].script_lock = self.script_lock
                devices[i].timepoint_barrier = self.timepoint_barrier

    def assign_script(self, script, location):
        """
        Assigns a script and dynamically creates a location-specific lock if needed.

        The creation of location-specific locks is not thread-safe and can lead
        to a race condition if multiple scripts for a new location are assigned
        concurrently.

        Args:
            script: The script object to execute.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
            # CRITICAL: This check-then-set block is a race condition.
            if not location in self.data_locks:
                lock = Lock()
                for dev in self.devices:
                    dev.data_locks[location] = lock
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating timepoints.

    In each timepoint, this thread spawns a pool of `ScriptThread`s (one for
    each script), waits for them to complete, and then synchronizes at a
    global barrier.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script_threads = []

    def run(self):
        """The main execution loop, organized into discrete timepoints."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Spawns one worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                script_thread = ScriptThread(self.device, script, location, \
                    neighbours)
                script_thread.start()
                self.script_threads.append(script_thread)

            # Reset for the next timepoint and wait for all workers to finish.
            self.device.timepoint_done.clear()
            for script_thread in self.script_threads:
                script_thread.join()
            self.script_threads = []
            
            # Invariant: All devices must synchronize at the barrier before the
            # next timepoint can begin.
            self.device.timepoint_barrier.wait()

class ScriptThread(Thread):
    """
    A worker thread that executes a single script, using a mixed-locking strategy.
    """

    def __init__(self, device, script, location, neighbours):
        """
        Initializes the ScriptThread.
        
        Args:
            device (Device): The parent device.
            script: The script to execute.
            location: The location to operate on.
            neighbours (list): List of neighbouring devices.
        """
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script using a nested locking scheme.
        """
        # Pre-condition: Acquire a location-specific lock to protect data reads.
        with self.device.data_locks[self.location]:
            script_data = []
            
            # Aggregate data from neighbours and self.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = self.script.run(script_data)

                # Pre-condition: Acquire a single global lock to protect data writes.
                # This serializes all write operations across the entire system.
                with self.device.script_lock:
                    # Broadcast the result to all participants.
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    
                    self.device.set_data(self.location, result)