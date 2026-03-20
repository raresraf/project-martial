"""
This module simulates a distributed network of devices that process sensor data.

This implementation uses a complex, manual approach to concurrency. Each `Device`
runs a master `DeviceThread`. For each time step, this master thread creates a
new, dedicated worker thread (`MyThread`) for every script it needs to execute.
It then manually manages the execution of these worker threads to limit the
number of concurrently active scripts to a fixed number (e.g., 8).

Synchronization is achieved using:
- `threading.Event` for signaling between the main thread and the device threads.
- A list of `threading.Lock` objects, one per location, to ensure exclusive
  access to location-specific data.
- A `ReusableBarrierSem` (imported from a missing `barrier.py` file) to
  synchronize all devices at the end of each time step.
"""

from threading import Event, Thread, Lock
# Note: The following import requires a 'barrier.py' file to be present.
from barrier import ReusableBarrierSem
class Device(object):
    """
    Represents a single device in the simulated network.

    Each device manages its own sensor data and executes assigned scripts in a
    dedicated master thread (`DeviceThread`).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.
        :param device_id: Unique identifier for the device.
        :param sensor_data: A dictionary of {location: data}.
        :param supervisor: An object that provides network topology (neighbours).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals that scripts for a time step are assigned.
        self.scripts = []
        self.timepoint_done = Event() # This Event appears unused in the current logic.
        self.devices = None
        self.barrier = None # Shared barrier to sync all devices.
        self.thread = DeviceThread(self)
        self.locations = [] # Shared list of location-specific locks.
        self.data_lock = Lock() # Lock for writing data.
        self.get_lock = Lock() # Lock for reading data.
        self.setup = Event() # Signals that the shared setup is complete.
        self.thread.start()
    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared concurrency primitives.

        This is intended to be called by a single "master" device (device_id 0).
        It creates the barrier and the location locks and shares them with all
        other devices in the simulation.
        """
        
        self.devices = devices
        barrier = ReusableBarrierSem(len(devices))
        if self.device_id == 0:
            # Create a hardcoded number of location locks.
            for _ in range(100):
                self.locations.append(Lock())
            # Distribute shared objects to all other devices.
            for dev in devices:
                dev.barrier = barrier
                dev.locations = self.locations
                dev.setup.set() # Signal other devices that setup is complete.
    def assign_script(self, script, location):
        """
        Assigns a script to run. A `None` script signals the end of assignments
        for the current time step.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set() # Signal the device thread to start processing.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.
        Note: Uses a separate lock from `set_data`, which can be dangerous
        if read/write atomicity is required.
        """
        with self.get_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location.
        Note: Uses a separate lock from `get_data`.
        """
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Joins the device's master thread to gracefully shut it down."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a single Device, managing the execution
    of scripts for each time step.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop, progressing in discrete time steps."""
        # Wait until the master device has finished setting up shared resources.
        self.device.setup.wait()
        while True:
            threads = []
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals end of simulation.
                break
            
            # Wait until all scripts for this time step have been assigned.
            self.device.script_received.wait()
            self.device.script_received.clear() # Reset for the next time step.

            # --- Manual Thread Management ---
            # Instead of a thread pool, create a new thread for each script.
            i = 0
            for _ in self.device.scripts:
                threads.append(MyThread(self.device, self.device.scripts, neighbours, i))
                i = i + 1
            
            scripts_rem = len(self.device.scripts)
            start = 0
            # Manually manage concurrency to run at most 8 threads at a time.
            if len(self.device.scripts) < 8:
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            else:
                # This logic implements a sliding window to run jobs in batches of 8.
                while True:
                    if scripts_rem == 0:
                        break
                    if scripts_rem >= 8:
                        for i in xrange(start, start + 8):
                            threads[i].start()
                        for i in xrange(start, start + 8):
                            threads[i].join()
                        start = start + 8
                        scripts_rem = scripts_rem - 8
                    else:
                        for i in xrange(start, start + scripts_rem):
                            threads[i].start()
                        for i in xrange(start, start + scripts_rem):
                            threads[i].join()
                        break
            
            # After all scripts for this device are done, wait at the global barrier.
            # This synchronizes all devices before the next time step.
            self.device.barrier.wait()

class MyThread(Thread):
    """A worker thread responsible for executing a single script."""
    def __init__(self, device, scripts, neighbours, indice):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours
        self.indice = indice

    def run(self):
        """The core logic for processing one script."""
        (script, location) = self.scripts[self.indice]

        # Acquire the lock for this specific location to ensure exclusive access.
        self.device.locations[location].acquire()
        
        # Gather data for the location from this device and its neighbours.
        script_data = []
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        if script_data:
            # Run the script on the collected data.
            result = script.run(script_data)
            # Broadcast the result to all involved devices.
            for device in self.neighbours:
                device.set_data(location, result)
                self.device.set_data(location, result)
        
        # Release the lock for this location.
        self.device.locations[location].release()
