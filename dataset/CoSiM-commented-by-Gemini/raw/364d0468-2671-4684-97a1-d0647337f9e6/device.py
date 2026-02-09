"""
A device simulation framework using a master-worker setup and batched execution.

This module implements a device simulation where one device (ID 0) creates and
distributes shared resources, including a barrier and a dictionary of
location-specific locks. The main device thread processes scripts in batches,
spawning a new thread for each script. A key feature is the locking strategy:
while script execution is parallelized by location, all underlying data reads
and writes are serialized by a single global lock, creating a bottleneck.
"""

from threading import Event, Thread, Lock
import barrier

class Device(object):
    """
    Represents a device that relies on a master device (ID 0) for the setup
    of shared synchronization primitives.
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
        self.scripts_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.data_lock = Lock()
        self.list_locks = {}
        self.barrier = None
        self.devices = None

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources using a master-device pattern.

        Device 0 discovers all sensor data locations to create a dictionary of
        locks and a shared barrier. Other devices then get references to these
        resources. The device threads are started before this setup is guaranteed
        to be complete for all devices, creating a potential race condition.
        
        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices

        if self.device_id == self.devices[0].device_id:
            self.barrier = barrier.ReusableBarrierCond(len(self.devices))
            # Block Logic: Master device discovers all locations and creates locks.
            for dev in self.devices:
                for location in dev.sensor_data:
                    self.list_locks[location] = Lock()
        else:
            # Worker devices get resources from the master device.
            self.barrier = devices[0].get_barrier()
            self.list_locks = devices[0].get_list_locks()
        
        # Threads are started immediately after setup, which can be racy.
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script for execution in the current timepoint.

        Args:
            script: The script object to execute.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set()


    def get_barrier(self):
        """Returns the shared barrier instance."""
        return self.barrier

    def get_list_locks(self):
        """Returns the shared dictionary of location-specific locks."""
        return self.list_locks

    def get_data(self, location):
        """
        Thread-safely retrieves sensor data for a specific location.
        All reads are serialized by a single global lock.
        """
        with self.data_lock:
            if location in self.sensor_data:
                data = self.sensor_data[location]
            else:
                data = None
        return data

    def set_data(self, location, data):
        """
        Thread-safely updates sensor data at a specific location.
        All writes are serialized by a single global lock.
        """
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, which orchestrates script execution
    by spawning worker threads in batches.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop, organized into discrete timepoints."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            # Block Logic: Spawns and joins worker threads in batches of 8.
            threads = []
            for (script, location) in self.device.scripts:
                threads.append(
                    ScriptThread(self.device, script, location, neighbours))
                if len(threads) == 8:
                    for thr in threads:
                        thr.start()
                    for thr in threads:
                        thr.join()
                    threads = []
            
            # Joins the final, smaller-than-8 batch of threads.
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()

            # Invariant: All devices must synchronize at the barrier before the
            # next timepoint can begin.
            self.device.barrier.wait()



class ScriptThread(Thread):
    """
    A worker thread that executes a single script.
    """

    def __init__(self, device, script, location, neighbours):
        """Initializes the ScriptThread."""
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script, using a location-specific lock for the overall
        operation and a global lock for each data access.
        """
        # Pre-condition: Acquire a lock specific to the data's location.
        self.device.list_locks[self.location].acquire()

        script_data = []

        # Aggregate data from neighbours. Each call to get_data is globally locked.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Get own data, also globally locked.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Invariant: Script only runs if there is data to process.
        if script_data != []:
            result = self.script.run(script_data)

            # Broadcast result. Each call to set_data is globally locked.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        self.device.list_locks[self.location].release()