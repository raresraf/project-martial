"""
This module defines a simulated Device that processes scripts in parallel
for a distributed sensor network simulation.

Each Device runs a main thread (`DeviceThread`) that dynamically spawns new
threads (`ScriptThread`) for each assigned script, up to a concurrent limit
controlled by a semaphore. Synchronization is managed via a class-level
barrier and instance-specific locks and events.
"""

from threading import Event, Thread, BoundedSemaphore, Lock
# Assuming cond_barrier.py contains the ReusableBarrier implementation.
from cond_barrier import ReusableBarrier


class Device(object):
    """
    Represents a single device in the simulated network.

    It receives scripts from a supervisor and processes them concurrently.
    """
    # --- Class-level synchronization primitives ---
    # A barrier shared by all Device instances to synchronize timepoints.
    barrier = None
    # An event to ensure the barrier is initialized before any device uses it.
    barrier_event = Event()

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor data.
            supervisor (Supervisor): The central supervisor controlling the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # --- Instance-level script management ---
        self.scripts = []  # A list to store (script, location) tuples.
        # Limits the number of concurrent script threads to 8.
        self.scripts_semaphore = BoundedSemaphore(8)
        self.scripts_lock = Lock()  # Protects access to the self.scripts list.

        # --- Instance-level synchronization events ---
        self.script_received = Event()  # Signals that a new script has been added.
        self.timepoint_done = Event()   # Signals that all scripts for a timepoint have been assigned.

        # The main thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared barrier for all devices.

        This should be called by one device (e.g., device 0) at the start
        of the simulation.
        """
        if Device.barrier is None and self.device_id == 0:
            Device.barrier = ReusableBarrier(len(devices))
            # Signal all devices that the barrier is ready.
            Device.barrier_event.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of assignment.

        Args:
            script (Script): The script object to execute.
            location (str): The location associated with the script.
        """
        with self.scripts_lock:
            self.script_received.set() # Signal that a script arrived (or assignment ended).
            if script is not None:
                self.scripts.append((script, location))
            else:
                # Sentinel script: signal that all scripts for this timepoint are received.
                self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates the sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main device thread to shut down gracefully."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a Device's lifecycle."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main lifecycle loop for the device.
        """
        # Wait until the shared barrier is initialized.
        Device.barrier_event.wait()
        while True:
            # Get current neighbors at the start of each timepoint.
            neighbours = self.device.supervisor.get_neighbours()

            # A `None` neighbor list is the signal to terminate.
            if neighbours is None:
                break

            script_index = 0
            script_threads = []
            while True:
                # --- Script Spawning Loop ---
                self.device.scripts_lock.acquire()
                if script_index < len(self.device.scripts):
                    # If there's an unhandled script, spawn a thread for it.
                    self.device.scripts_lock.release()
                    # Block if there are already 8 active script threads.
                    self.device.scripts_semaphore.acquire()

                    # Create and start a new thread for the script.
                    script, location = self.device.scripts[script_index]
                    thread = ScriptThread(self.device, script, location, neighbours)
                    script_threads.append(thread)
                    thread.start()

                    script_index += 1
                else:
                    # --- Wait Condition ---
                    # If all assigned scripts are running AND the supervisor signaled
                    # the end of the timepoint, we can break and wait for threads to finish.
                    if self.device.timepoint_done.is_set() and script_index == len(self.device.scripts):
                        self.device.timepoint_done.clear()
                        self.device.scripts = [] # Clear scripts for next round.
                        self.device.scripts_lock.release()
                        break
                    else:
                        # Otherwise, wait for a new script to be assigned.
                        self.device.scripts_lock.release()
                        self.device.script_received.wait()
                        self.device.script_received.clear() # Consume the event.

            # Wait for all spawned script threads for this timepoint to complete.
            for script_thread in script_threads:
                script_thread.join()

            # Synchronize with all other devices before starting the next timepoint.
            Device.barrier.wait()


class ScriptThread(Thread):
    """A worker thread that executes a single script."""
    # Class-level dictionary of locks to ensure atomic access to locations across all threads.
    locations_locks = {}

    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.location = location
        self.script = script
        self.device = device
        self.neighbours = neighbours

        # Create a lock for the location if it doesn't exist.
        if location not in ScriptThread.locations_locks:
            ScriptThread.locations_locks[location] = Lock()

    def run(self):
        """
        Executes the script, handling data aggregation and propagation.
        """
        # Use a `with` statement for automatic lock acquisition and release.
        with ScriptThread.locations_locks[self.location]:
            # --- Data Aggregation ---
            script_data = []
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # --- Script Execution and Data Propagation ---
            if script_data:
                result = self.script.run(script_data)
                # Update data on neighbors and the local device.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)
        
        # Release the semaphore slot, allowing another ScriptThread to run.
        self.device.scripts_semaphore.release()
