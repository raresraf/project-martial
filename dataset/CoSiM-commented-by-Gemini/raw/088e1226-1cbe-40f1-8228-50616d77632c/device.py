"""
This module defines a simulated distributed device network.

The architecture follows a Bulk Synchronous Parallel (BSP) model, coordinated
by a reusable barrier and fine-grained locking on data locations. It appears
to be designed for a simulation where devices operate in discrete, synchronized
timesteps.
"""

from threading import Event, Thread, Lock
# Note: 'barrier' is a custom module, assumed to contain a ReusableBarrier class.
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single device in the network simulation.

    Each device runs its own thread, processes assigned scripts, and synchronizes
    with other devices using a shared barrier and location-specific locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device and starts its main control thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that scripts for the current timepoint have been assigned.
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # This event seems unused in the current logic.
        self.devices = None
        self.barrier = None
        self.thread = DeviceThread(self)
        # A shared list of locks, one for each data location.
        self.locations = []
        # Locks to ensure thread-safe access to sensor_data dictionary.
        self.data_lock = Lock()
        self.get_lock = Lock()
        # Event to signal that the initial setup by the master device is complete.
        self.setup = Event()
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for the entire device network.

        This method implements a centralized setup pattern where the device with ID 0
        creates and distributes the shared barrier and location locks to all devices.
        """
        self.devices = devices
        barrier = ReusableBarrierSem(len(devices))
        if self.device_id == 0:
            # The master device creates a pool of location-based locks.
            for _ in range(100):
                self.locations.append(Lock())

            # Distribute the shared barrier and locks to all devices.
            for dev in devices:
                dev.barrier = barrier
                dev.locations = self.locations
                # Signal to other devices that setup is complete.
                dev.setup.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of assignments.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If the script is None, it signals that all scripts for the
            # current timepoint have been sent, waking up the DeviceThread.
            self.script_received.set()

    def get_data(self, location):
        """
        Thread-safely retrieves data from a specific sensor location.
        """
        with self.get_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Thread-safely updates data at a specific sensor location.
        """
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Waits for the device's main thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control-loop thread for a device, orchestrating its lifecycle in
    synchronized time steps.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Implements the main time-stepped execution loop (BSP model).
        """
        # Wait until the master device has finished initialization.
        self.device.setup.wait()
        while True:
            threads = []
            neighbours = self.device.supervisor.get_neighbours()
            # A `None` from the supervisor indicates the simulation is over.
            if neighbours is None:
                break

            # Wait for the signal that all scripts for this step are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Create worker threads for each assigned script.
            i = 0
            for _ in self.device.scripts:
                threads.append(MyThread(self.device, self.device.scripts, neighbours, i))
                i = i + 1
            
            # --- Parallel Execution with Throttling ---
            # Run the script-executing threads in batches of 8.
            scripts_rem = len(self.device.scripts)
            start = 0
            if len(self.device.scripts) < 8:
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            else:
                while True:
                    if scripts_rem == 0:
                        break
                    # Process a full batch of 8 threads.
                    if scripts_rem >= 8:
                        for i in xrange(start, start + 8):
                            threads[i].start()
                        for i in xrange(start, start + 8):
                            threads[i].join()
                        start = start + 8
                        scripts_rem = scripts_rem - 8
                    # Process the final, smaller batch.
                    else:
                        for i in xrange(start, start + scripts_rem):
                            threads[i].start()
                        for i in xrange(start, start + scripts_rem):
                            threads[i].join()
                        break
            
            # --- Global Barrier ---
            # Wait here until all devices in the network have finished their
            # computation for the current time step.
            self.device.barrier.wait()

class MyThread(Thread):
    """
    A worker thread to execute one script for one location.
    """
    def __init__(self, device, scripts, neighbours, indice):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours
        self.indice = indice

    def run(self):
        """
        Executes a script, using a location-specific lock to ensure data consistency.
        """
        (script, location) = self.scripts[self.indice]
        # --- Fine-Grained Locking ---
        # Acquire a lock for the specific data location being processed. This allows
        # other MyThread workers to process different locations in parallel.
        self.device.locations[location].acquire()

        # Gather data from neighbors and self for the given location.
        script_data = []
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        # If there is data, run the script and propagate the result.
        if script_data != []:
            result = script.run(script_data)
            # Update the data on this device and all its neighbors.
            for device in self.neighbours:
                device.set_data(location, result)
                self.device.set_data(location, result)
        
        # Release the lock for this location.
        self.device.locations[location].release()
