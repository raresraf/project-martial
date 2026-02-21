


"""
Models a device in a highly concurrent, distributed simulation framework.

This module defines a complex architecture for simulating device interactions.
It features a `Device` class that manages state and synchronization, a main
`DeviceThread` that orchestrates time steps, and a pool of `ScriptWorker`
threads that execute computational tasks in parallel. Synchronization across
devices is managed by a global barrier and a set of shared condition variables,
one for each data location, to prevent race conditions during script execution.
"""

from threading import Event, Thread, Lock, Condition
from reusable_barrier import ReusableBarrier

NUM_THREADS = 8  # The size of the script execution worker pool for each device.


class Device(object):
    """Represents a device, holding data and managing worker threads.

    This class is the central hub for a single device, owning the sensor data,
    the script queue, and all necessary synchronization primitives. It spawns
    a main `DeviceThread` and a pool of `ScriptWorker` threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device."""
        self.device_id = device_id
        self.supervisor = supervisor

        # Event to signal that the device is ready after initial setup.
        self.ready_to_start = Event()

        # Locks and data
        self.data_lock = Lock()
        self.sensor_data = sensor_data
        self.location_busy = {location: False for location in self.sensor_data}
        self.location_busy_lock = Lock()

        # State for managing scripts within a time step.
        self.scripts = []
        self.scripts_assigned = False
        self.scripts_enabled = False
        self.scripts_started_idx = 0
        self.scripts_done_idx = 0

        # Condition variables for coordinating between the DeviceThread (producer)
        # and ScriptWorkers (consumers).
        self.scripts_lock = Lock()
        self.scripts_condition = Condition(self.scripts_lock)
        self.scripts_done_condition = Condition(self.scripts_lock)

        # Thread management
        self.thread_running = True
        self.thread = DeviceThread(self)
        self.worker_threads = [ScriptWorker(self, i) for i in range(NUM_THREADS)]

        for thread in self.worker_threads:
            thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared synchronization objects.

        The device with ID 0 acts as the master, creating a global timestep
        barrier and a dictionary of shared `Condition` variables (one for each
        unique data location). These are then distributed to all other devices.
        """
        if self.device_id == 0:
            timestep_barrier = ReusableBarrier(len(devices))
            location_conditions = {}

            # Create one shared Condition object for each unique data location.
            for device in devices:
                for location in device.sensor_data:
                    if location not in location_conditions:
                        location_conditions[location] = Condition()

            # Distribute the shared objects to all devices.
            for device in devices:
                device.location_conditions = location_conditions
                device.timestep_barrier = timestep_barrier
            
            for device in devices:
                device.ready_to_start.set()

        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be processed in the current time step."""
        with self.scripts_lock:
            if script is not None:
                self.scripts.append((script, location))
                # Notify worker threads that a new script is available.
                self.scripts_condition.notify_all()
            else:
                # A None script marks the end of script assignment for this step.
                self.scripts_assigned = True
                # Notify the main DeviceThread that assignments are done.
                self.scripts_done_condition.notify_all()

    def is_busy(self, location):
        """Checks if a given data location is currently being processed."""
        with self.location_busy_lock:
            return location in self.location_busy and self.location_busy[location]

    def set_busy(self, location, value):
        """Sets the busy status for a given data location."""
        with self.location_busy_lock:
            self.location_busy[location] = value

    def has_data(self, location):
        """Checks if the device contains data for a given location."""
        with self.data_lock:
            return location in self.sensor_data

    def get_data(self, location):
        """Gets data from a specific sensor location."""
        with self.data_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets data at a specific sensor location."""
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Joins all threads for a clean shutdown."""
        self.thread.join()
        for thread in self.worker_threads:
            thread.join()


class ScriptWorker(Thread):
    """A worker thread that executes scripts from the device's script list."""
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
                # Wait until the main DeviceThread enables script processing for the timestep
                # and there are un-started scripts available.
                while not self.device.scripts_enabled or \
                        self.device.scripts_started_idx >= len(self.device.scripts):
                    self.condition.wait()
                    if not self.device.thread_running:
                        return

                # Atomically claim a script from the list.
                script_tuple = self.device.scripts[self.device.scripts_started_idx]
                self.device.scripts_started_idx += 1
                self.condition.notify_all()

                # Release the main lock to run the script, allowing other workers
                # to claim other scripts in parallel.
                self.lock.release()
                self.run_script(script_tuple[0], script_tuple[1])
                self.lock.acquire()

                # Atomically increment the done counter and notify the main DeviceThread.
                self.device.scripts_done_idx += 1
                self.done_condition.notify_all()

    def run_script(self, script, location):
        """Executes a single script, handling complex cross-device synchronization."""
        # Acquire the lock associated with the specific data location.
        with self.device.location_conditions[location]:
            # Identify all devices (self and neighbors) that have relevant data.
            script_devices = [
                device for device in self.device.neighbours if device.has_data(location)
            ]
            if self.device.has_data(location):
                script_devices.append(self.device)

            if not script_devices:
                return

            # Wait until the location is not busy on any of the relevant devices.
            # This is a form of distributed lock.
            while any(device.is_busy(location) for device in script_devices):
                self.device.location_conditions[location].wait()

            # Mark the location as busy on all relevant devices.
            for device in script_devices:
                device.set_busy(location, True)
            self.device.location_conditions[location].notify_all()

            # Gather data now that we have exclusive access.
            script_data = [device.get_data(location) for device in script_devices]

            # Run the computation. The location condition lock is held to prevent
            # other operations but the main device data lock is not.
            result = script.run(script_data)

            # Write the result back and release the busy flags.
            for device in script_devices:
                device.set_data(location, result)
                device.set_busy(location, False)
            # Notify any other workers waiting on this location that it's free now.
            self.device.location_conditions[location].notify_all()


class DeviceThread(Thread):
    """The main supervising thread for a device's lifecycle."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Orchestrates the device's participation in synchronized time steps."""
        self.device.ready_to_start.wait()

        while True:
            # All devices wait here, ensuring time steps start simultaneously.
            self.device.timestep_barrier.wait()

            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                break  # Supervisor signals shutdown.

            # --- Producer Logic ---
            # Reset state for the new time step and enable worker threads.
            with self.device.scripts_lock:
                self.device.scripts_started_idx = 0
                self.device.scripts_done_idx = 0
                self.device.scripts_enabled = True
                # Wake up any waiting worker threads.
                self.device.scripts_condition.notify_all()

            # --- Wait for Consumers ---
            # Wait until all scripts assigned for this step are processed by the workers.
            with self.device.scripts_lock:
                while not self.device.scripts_assigned or \
                      self.device.scripts_done_idx < len(self.device.scripts):
                    self.device.scripts_done_condition.wait()
                self.device.scripts_enabled = False

        # --- Shutdown ---
        # Signal all worker threads to terminate.
        with self.device.scripts_lock:
            self.device.thread_running = False
            self.device.scripts_condition.notify_all()
