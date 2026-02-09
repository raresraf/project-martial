"""
A device simulation framework using a thread-per-script model and nested locks.

This module defines a system where each device, in each time step, spawns a
dedicated 'helper' thread for every script it needs to execute. Synchronization
is attempted through dynamically created, nested locks—one for the script's
location and another for the script itself. This nested locking introduces a
risk of deadlock, and the dynamic creation of locks is not thread-safe.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    """
    Represents a device that manages a pool of script-executing helper threads.
    """

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
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

        self.location_locks = {} 
        self.script_locks = {}
        self.barrier = None

        self.threads = []
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier to all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        barrier = ReusableBarrierSem(len(devices))
        for device in devices:
            device.barrier = barrier

    def assign_script(self, script, location):
        """
        Receives a script from the supervisor for the current timepoint.

        Args:
            script: The script object to execute.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread and any helper threads to complete."""
        self.thread.join()
        for thread in self.threads:
            thread.join()


class DeviceThreadHelper(Thread):
    """
    A helper thread responsible for executing a single script.

    This thread attempts to manage concurrency by using a nested locking scheme,
    first acquiring a lock for the data's location, and then a lock for the
    script itself. The locks are created dynamically in a non-thread-safe manner.
    """

    def __init__(self, device, script, location, neighbours):
        """
        Initializes the helper thread.

        Args:
            device (Device): The parent device.
            script: The script to execute.
            location: The location to operate on.
            neighbours (list): List of neighbouring devices.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script


    def run(self):
        """The main execution logic for processing one script."""
        script_data = []
        
        # CRITICAL: This block creates locks dynamically without any
        # synchronization, which can lead to a race condition where multiple
        # threads attempt to create the same lock simultaneously.
        if self.location not in self.device.location_locks:
            self.device.location_locks[self.location] = Lock()

        if self.script not in self.device.script_locks:
            self.device.script_locks[self.script] = Lock()

        # Pre-condition: Acquire the lock for the specific data location.
        self.device.location_locks[self.location].acquire()

        # Aggregate data from all neighbours (including self).
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        if script_data != []:
            # Pre-condition: Acquire a second, nested lock for the script.
            # This nested locking can lead to deadlocks if different threads
            # acquire the same locks in a different order.
            self.device.script_locks[self.script].acquire()
            result = self.script.run(script_data)
            self.device.script_locks[self.script].release()

            # Broadcast the result while still holding the location lock.
            for device in self.neighbours:
                device.set_data(self.location, result)

        self.device.location_locks[self.location].release()


class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating timepoints.

    For each timepoint, this thread spawns a new helper thread for every
    assigned script, waits for them all to complete, and then synchronizes
    with other devices at a global barrier.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop, organized into discrete timepoints."""
        while True:
            # Get neighbours for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break 

            # The device adds itself to the list of neighbours to streamline
            # data gathering in the helper threads.
            neighbours.append(self.device) 

            # Wait until the supervisor has finished assigning all scripts.
            self.device.timepoint_done.wait() 

            # Block Logic: Spawns one helper thread per assigned script.
            for (script, location) in self.device.scripts:
                thread = DeviceThreadHelper(self.device, script, location, neighbours)
                self.device.threads.append(thread)
                thread.start()

            # Wait for all helper threads for this timepoint to complete.
            for thread in self.device.threads:
                thread.join()

            self.device.threads = []

            # Invariant: Synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()