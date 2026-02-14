"""
This module defines a sophisticated framework for a distributed device simulation.

It uses a hierarchical threading and synchronization model to manage concurrent
script execution across a network of devices in discrete time steps.

The main classes are:
- Device: Represents a node in the network. It encapsulates state, data, and
  manages a pool of worker threads and a main control thread.
- ScriptWorker: A worker thread that processes scripts. These workers use a
  work-stealing pattern from a shared script list.
- DeviceThread: The main control thread for a single device, responsible for
  orchestrating the device's lifecycle through synchronized time steps.
"""

from threading import Event, Thread, Lock, Condition
from reusable_barrier import ReusableBarrier

NUM_THREADS = 8


class Device(object):
    """
    Represents a single device in the simulated network.

    This class manages the device's local data, its assigned scripts, and a
    pool of worker threads (`ScriptWorker`) to execute them. It relies on a
    complex set of synchronization primitives to coordinate with other devices
    and its own internal threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (Supervisor): The central supervisor for the simulation.
        """
        self.device_id = device_id
        self.supervisor = supervisor
        self.sensor_data = sensor_data
        
        # --- State and Synchronization Primitives ---
        self.ready_to_start = Event()  # Signals that global setup is complete.
        self.data_lock = Lock()
        
        # Tracks if a data location is being operated on to prevent race conditions.
        self.location_busy = {location: False for location in self.sensor_data}
        self.location_busy_lock = Lock()

        # --- Script Management ---
        self.scripts = []
        self.scripts_assigned = False # Flag: True when all scripts for a time step are assigned.
        self.scripts_enabled = False  # Flag: True to allow workers to process scripts.
        self.scripts_started_idx = 0  # Index for the work-stealing queue.
        self.scripts_done_idx = 0     # Counter for completed scripts.

        # Condition variables for coordinating script execution between the
        # DeviceThread and the ScriptWorkers.
        self.scripts_lock = Lock()
        self.scripts_condition = Condition(self.scripts_lock)
        self.scripts_done_condition = Condition(self.scripts_lock)

        # --- Thread Management ---
        self.thread_running = True
        self.thread = DeviceThread(self)  # Main control thread for this device.
        self.worker_threads = [ScriptWorker(self, i) for i in range(NUM_THREADS)]

        for thread in self.worker_threads:
            thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global synchronization objects for all devices in the simulation.

        This must be called on the root device (device_id == 0). It creates and
        distributes a single global timestep barrier and a shared dictionary
        of condition variables for location-based locking.
        """
        if self.device_id == 0:
            timestep_barrier = ReusableBarrier(len(devices))
            location_conditions = {}

            # Create a condition variable for each unique data location across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if location not in location_conditions:
                        location_conditions[location] = Condition()

            # Distribute the shared objects to all devices.
            for device in devices:
                device.location_conditions = location_conditions
                device.timestep_barrier = timestep_barrier
                device.ready_to_start.set()  # Signal that devices can start their main loop.

        self.thread.start()

    def assign_script(self, script, location):
        """

        Assigns a script to the device. Called by the supervisor.

        Args:
            script (Script): The script to execute. If None, it signals that
                             all scripts for the current time step have been assigned.
            location (any): The data location for the script.
        """
        with self.scripts_lock:
            if script is not None:
                self.scripts.append((script, location))
                self.scripts_condition.notify_all()  # Wake up a worker.
            else:
                self.scripts_assigned = True
                self.scripts_done_condition.notify_all()  # Wake up the main DeviceThread.

    def is_busy(self, location):
        """Checks if a location is currently being processed."""
        with self.location_busy_lock:
            return self.location_busy.get(location, False)

    def set_busy(self, location, value):
        """Sets the busy status of a location."""
        with self.location_busy_lock:
            self.location_busy[location] = value

    def has_data(self, location):
        """Checks if the device contains data for a given location."""
        with self.data_lock:
            return location in self.sensor_data

    def get_data(self, location):
        """Thread-safely gets data from a specific location."""
        with self.data_lock:
            return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Thread-safely sets data at a specific location."""
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control and worker threads."""
        self.thread.join()
        for thread in self.worker_threads:
            thread.join()


class ScriptWorker(Thread):
    """
    A worker thread that executes scripts for a device.

    Workers for a device share a list of scripts and use a work-stealing
    pattern to process them in parallel.
    """
    def __init__(self, device, index):
        Thread.__init__(self, name="Worker thread %d for device %d" % (index, device.device_id))
        self.device = device
        self.lock = device.scripts_lock
        self.done_condition = device.scripts_done_condition
        self.condition = device.scripts_condition

    def run(self):
        """Main loop for the worker thread."""
        with self.lock:
            while self.device.thread_running:
                # Wait until scripts are enabled and there are un-started scripts.
                while self.device.thread_running and (not self.device.scripts_enabled or \
                       self.device.scripts_started_idx >= len(self.device.scripts)):
                    self.condition.wait()

                if not self.device.thread_running:
                    break

                # Work-stealing: Atomically get the next script to run.
                script_tuple = self.device.scripts[self.device.scripts_started_idx]
                self.device.scripts_started_idx += 1
                self.condition.notify_all() # Notify others in case they can now work.

                # Release the lock while running the script to allow other workers to steal.
                self.lock.release()
                self.run_script(script_tuple[0], script_tuple[1])
                self.lock.acquire()

                # Atomically report that a script is done.
                self.device.scripts_done_idx += 1
                self.done_condition.notify_all() # Notify the main DeviceThread.

    def run_script(self, script, location):
        """
        Executes a single script with complex location-based synchronization.
        """
        # Acquire the condition variable for this specific data location.
        with self.device.location_conditions[location] as loc_cond:
            # Identify all devices involved in this computation.
            script_devices = [
                dev for dev in self.device.neighbours if dev.has_data(location)
            ]
            if self.device.has_data(location):
                script_devices.append(self.device)

            if not script_devices:
                return

            # Wait until the location is free on all participating devices.
            while any(dev.is_busy(location) for dev in script_devices):
                loc_cond.wait()

            # Mark the location as busy on all participating devices.
            for device in script_devices:
                device.set_busy(location, True)
            loc_cond.notify_all() # Notify others that the state has changed.

            # Get data from all devices.
            script_data = [dev.get_data(location) for dev in script_devices]

            # --- Critical section finished, release lock to compute ---
            loc_cond.release()
            result = script.run(script_data)
            loc_cond.acquire()
            # --- Re-acquired lock to write results ---

            # Write results back and mark the location as free.
            for device in script_devices:
                device.set_data(location, result)
                device.set_busy(location, False)
            loc_cond.notify_all() # Notify waiting threads that location is free.


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    It orchestrates the device's participation in the global simulation
    time steps and manages its pool of ScriptWorkers.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main lifecycle of a device, stepping through time."""
        # Wait for the global setup to be completed by the root device.
        self.device.ready_to_start.wait()

        while True:
            # --- Global Time Step Synchronization ---
            # All DeviceThreads wait here for the start of the next time step.
            self.device.timestep_barrier.wait()

            # The supervisor signals the end of the simulation by returning None.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                break

            # --- Enable and Wait for Local Workers ---
            with self.device.scripts_lock:
                # Reset script counters for the new time step.
                self.device.scripts_started_idx = 0
                self.device.scripts_done_idx = 0
                # Allow worker threads to start processing scripts.
                self.device.scripts_enabled = True
                self.device.scripts_condition.notify_all()

            # Wait until the supervisor has assigned all scripts AND
            # all assigned scripts have been completed by the workers.
            with self.device.scripts_lock:
                while not self.device.scripts_assigned or \
                      self.device.scripts_done_idx < len(self.device.scripts):
                    self.device.scripts_done_condition.wait()
                # Disable script processing until the next time step.
                self.device.scripts_enabled = False

        # --- Shutdown ---
        # Signal all worker threads to terminate.
        with self.device.scripts_lock:
            self.device.thread_running = False
            self.device.scripts_condition.notify_all()
