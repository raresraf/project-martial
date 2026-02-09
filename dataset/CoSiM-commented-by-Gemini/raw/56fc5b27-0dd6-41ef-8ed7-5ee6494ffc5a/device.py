"""
@file device.py
@brief Defines a device model for a simulation with concurrency-limited script execution.

This file implements a `Device` class that uses a `Semaphore` to limit the
number of concurrently running script threads to a maximum of 8. It employs a
two-phase barrier for synchronization and dynamically creates and distributes
location-specific locks before executing scripts in each timepoint.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem

class Device(object):
    """
    Represents a device in a simulation, managing sensor data, script execution,
    and a semaphore for limiting concurrent worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): A dictionary of the device's sensor readings.
            supervisor: The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.barrier = None
        self.threads = []
        # A semaphore to limit the number of active script threads to 8.
        self.semaphore = Semaphore(8)
        self.lock = {}
        self.all_devices = []
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the list of all devices and the shared synchronization barrier.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if len(self.all_devices) == 0:
            self.all_devices = devices

        # Block Logic: The first device to enter will create and distribute the barrier.
        if self.barrier is None:
            barrier = ReusableBarrierSem(len(devices))
            for device in devices:
                device.barrier = barrier

    def update_locks(self, scripts):
        """
        Dynamically creates and distributes locks for script locations.

        For each location in the assigned scripts, if a lock does not already
        exist, it is created and the same lock instance is propagated to all
        other devices in the simulation.

        Args:
            scripts (list): The list of (script, location) tuples for the timepoint.
        """
        for (_, location) in scripts:
            if not self.lock.has_key(location):
                self.lock[location] = Lock()
                # Invariant: Ensure all devices share the exact same lock instance for a location.
                for device in self.all_devices:
                    device.lock[location] = self.lock[location]

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script: The script object to run.
            location: The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of script assignment for the timepoint.
            self.script_received.set()

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

class DeviceThread(Thread):
    """
    The main control thread for a Device, managing the two-phase barrier
    synchronization and semaphore-limited script execution.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            
            # Ensure locks for all script locations are created and distributed.
            self.device.update_locks(self.device.scripts)

            # --- First Barrier ---
            # All devices wait here to ensure locks are set up everywhere before proceeding.
            self.device.barrier.wait()

            # Block Logic: Start a worker thread for each script, respecting the semaphore limit.
            for script in self.device.scripts:
                thread = ScriptThread(self.device, neighbours, script)
                self.device.threads.append(thread)

                # Acquire semaphore before starting a new thread to limit concurrency.
                self.device.semaphore.acquire()
                thread.start()

            # Wait for all started script threads to complete.
            for thread in self.device.threads:
                thread.join()

            self.device.threads = []
            self.device.script_received.clear()

            # --- Second Barrier ---
            # Wait for all devices to finish their script execution for this timepoint.
            self.device.barrier.wait()

class ScriptThread(Thread):
    """
    A worker thread for executing one script.
    
    It handles acquiring the location-specific lock and releasing the device's
    master semaphore upon completion.
    """
    def __init__(self, device, neighbours, script):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script

    def run(self):
        # Pre-condition: Acquire the specific lock for the script's location.
        self.device.lock.get(self.script[1]).acquire()

        script_data = []
        
        # Block Logic: Gather data from neighbors and the local device.
        for device in self.neighbours:
            data = device.get_data(self.script[1])
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(self.script[1])
        if data is not None:
            script_data.append(data)

        # Invariant: Data is gathered, and the script is ready to run.
        if script_data != []:
            result = self.script[0].run(script_data)

            # Propagate the result back to all relevant devices.
            for device in self.neighbours:
                device.set_data(self.script[1], result)
                
            self.device.set_data(self.script[1], result)

        # Release the location-specific lock.
        self.device.lock.get(self.script[1]).release()

        # Release the semaphore to allow another script thread to be started.
        self.device.semaphore.release()