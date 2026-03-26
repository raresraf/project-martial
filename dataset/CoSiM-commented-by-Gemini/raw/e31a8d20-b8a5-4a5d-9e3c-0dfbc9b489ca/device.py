"""
device.py

@brief A distributed device simulation using a thread-per-task execution model within
       a Bulk Synchronous Parallel (BSP) framework.
@description This module simulates a network of devices that process scripts in synchronized
supersteps (timepoints). Each device runs a master control thread (`DeviceThread`) that,
for each timepoint, spawns a new, separate thread (`RunScript`) for every script to be
executed. Global synchronization between devices is managed by a reusable barrier.
"""

from threading import Event, Thread, Semaphore, Lock, RLock
from reusable_barrier import ReusableBarrier # Assumes a custom ReusableBarrier implementation.

class Device(object):
    """
    Represents a single device (node) in the distributed network.

    Manages its own data, scripts, and the lifecycle of its master control thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping locations to sensor values.
            supervisor (object): A supervisor object that provides network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        
        # --- Synchronization Primitives ---
        # A shared barrier to synchronize all devices at the start/end of a computation phase.
        self.barrier = None
        # A dictionary of re-entrant locks for each data location, shared across all devices.
        self.dislocksdict = None
        # A per-device lock, likely intended to protect its own sensor_data.
        self.lock = None
        # Semaphores used for a complex, one-time setup of shared resources.
        self.sem = Semaphore(1)
        self.sem2 = Semaphore(0)
        # Event to signal that all scripts for a timepoint have been assigned.
        self.timepoint_done = Event()

        # Each device has one master control thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects (barrier, locks)
        to all devices in the simulation. This complex setup is orchestrated by
        device 0.
        """
        all_locations = [loc for dev in devices for loc in dev.sensor_data]
        
        # Block Logic: Device 0 is responsible for creating the shared objects.
        if self.device_id == 0:
            self.sem2.release()
            self.barrier = ReusableBarrier(len(devices))
            self.dislocksdict = {loc: RLock() for loc in set(all_locations)}
            self.lock = Lock()

        # All devices wait here until device 0 has created the shared objects.
        self.sem2.acquire()

        # Block Logic: All other devices copy the references to the shared objects.
        for d in devices:
            if d.barrier is None:
                d.barrier = self.barrier
                d.sem2.release()
                d.dislocksdict = self.dislocksdict
                d.lock = Lock()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of
        assignments for the current timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location, or None if not present."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the master device thread to complete."""
        self.thread.join()


class RunScript(Thread):
    """
    A one-off thread created to execute a single script.
    """
    def __init__(self, script, location, neighbours, device):
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        Executes the script logic: acquires location lock, gathers data, runs the
        script, and writes back the results.
        """
        # Acquire the re-entrant lock for the specific data location. This ensures
        # that only one script is operating on this location across the entire system.
        self.device.dislocksdict[self.location].acquire()
        
        # --- Data Gathering Phase ---
        script_data = []
        for device in self.neighbours:
            # The per-device lock is acquired just for the get_data call.
            with device.lock:
                data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        with self.device.lock:
            data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # --- Computation Phase ---
        if script_data:
            result = self.script.run(script_data)
            
            # --- Write-Back Phase ---
            # Update the result on all neighbors and the local device.
            for device in self.neighbours:
                with device.lock:
                    device.set_data(self.location, result)
            with self.device.lock:
                self.device.set_data(self.location, result)
        
        self.device.dislocksdict[self.location].release()


class DeviceThread(Thread):
    """
    The master control thread for a single Device, orchestrating the BSP supersteps.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop, processing one superstep per iteration."""
        while True:
            # --- Superstep Start ---
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # Supervisor signals shutdown.

            # Wait for supervisor to signal that all scripts for this step are assigned.
            self.device.timepoint_done.wait()
            
            # --- Synchronization 1 ---
            # All devices wait here before starting computation.
            self.device.barrier.wait()
            
            # --- Computation Phase ---
            # Inefficiently create a new thread for every script.
            script_threads = [RunScript(s, l, neighbours, self.device) for s, l in self.device.scripts]
            
            for t in script_threads:
                t.start()
            for t in script_threads:
                t.join()  # Wait for all script threads for this device to complete.

            # --- Synchronization 2 ---
            # All devices wait here after computation is finished.
            self.device.barrier.wait()
            
            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()
