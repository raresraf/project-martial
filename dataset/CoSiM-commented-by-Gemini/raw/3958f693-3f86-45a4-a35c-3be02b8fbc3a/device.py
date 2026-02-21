"""
A simulation framework for distributed devices using complex synchronization.

This module implements a distributed device simulation using a highly manual and
intricate set of synchronization primitives. It features a custom producer-consumer
pattern managed with `threading.Condition` variables instead of a standard queue,
and a complex distributed locking scheme for data access based on location.
The simulation proceeds in synchronized time steps, coordinated by a reusable barrier.
"""

from threading import Event, Thread, Lock, Condition
from reusable_barrier import ReusableBarrier

NUM_THREADS = 8

class Device(object):
    """Manages the state and threads for a single simulated device.

    This class acts as a central hub for a device, holding its data, managing a
    pool of worker threads, and handling the complex shared state required for
    synchronization between workers and other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its associated threads and locks."""
        self.device_id = device_id
        self.supervisor = supervisor

        # Event to signal that the device is ready after initial setup.
        self.ready_to_start = Event()

        # Lock for thread-safe access to this device's own sensor_data.
        self.data_lock = Lock()
        self.sensor_data = sensor_data

        # A dictionary and lock to manage a 'busy' state for data locations.
        # This is part of a complex distributed locking scheme.
        self.location_busy = {location: False for location in self.sensor_data}
        self.location_busy_lock = Lock()

        # The 'queue' of scripts is a simple list, manually managed with indexes.
        self.scripts = []
        self.scripts_assigned = False
        self.scripts_enabled = False
        self.scripts_started_idx = 0
        self.scripts_done_idx = 0

        # A Condition variable is used to coordinate between the DeviceThread
        # (producer) and the ScriptWorkers (consumers).
        self.scripts_lock = Lock()
        self.scripts_condition = Condition(self.scripts_lock)
        self.scripts_done_condition = Condition(self.scripts_lock)

        # Create and start the device's main thread and worker pool.
        self.thread_running = True
        self.thread = DeviceThread(self)
        self.worker_threads = [ScriptWorker(self, i) for i in range(NUM_THREADS)]

        for thread in self.worker_threads:
            thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes shared resources for the entire device group.

        Device 0 acts as the master, creating a shared timestep barrier and a
        set of shared, location-based condition variables for all devices.
        """
        if self.device_id == 0:
            timestep_barrier = ReusableBarrier(len(devices))
            location_conditions = {}

            # Create one Condition variable for each unique data location.
            for device in devices:
                for location in device.sensor_data:
                    if not location in location_conditions:
                        location_conditions[location] = Condition()

            # Distribute the shared resources to all devices.
            for device in devices:
                device.location_conditions = location_conditions
                device.timestep_barrier = timestep_barrier

            # Signal all devices that setup is complete.
            for device in devices:
                device.ready_to_start.set()

        self.thread.start()

    def assign_script(self, script, location):
        """Adds a script to the device's workload for the current timestep."""
        if script is not None:
            with self.scripts_lock:
                self.scripts.append((script, location))
                self.scripts_condition.notify_all()
        else:
            # A None script signifies the end of script assignment for the step.
            with self.scripts_lock:
                self.scripts_assigned = True
                self.scripts_done_condition.notify_all()

    def is_busy(self, location):
        """Checks if the device is busy with a computation at a location."""
        with self.location_busy_lock:
            return location in self.location_busy and self.location_busy[location]

    def set_busy(self, location, value):
        """Sets the busy status for a given location."""
        with self.location_busy_lock:
            self.location_busy[location] = value

    def has_data(self, location):
        """Checks if the device contains data for a given location."""
        with self.data_lock:
            return location in self.sensor_data

    def get_data(self, location):
        """Gets data from a location in a thread-safe manner."""
        with self.data_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets data at a location in a thread-safe manner."""
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main and worker threads for a graceful shutdown."""
        self.thread.join()
        for thread in self.worker_threads:
            thread.join()

class ScriptWorker(Thread):
    """A worker thread that consumes and executes scripts for a Device."""
    def __init__(self, device, index):
        Thread.__init__(self, name="Worker thread %d for device %d" % (index, device.device_id))
        self.device = device
        self.lock = device.scripts_lock
        self.done_condition = device.scripts_done_condition
        self.condition = device.scripts_condition

    def run(self):
        """The main loop for the worker thread.

        Implements a consumer in a custom producer-consumer pattern. It waits
        on a condition variable until scripts are available, processes one
        script, and repeats.
        """
        with self.lock:
            while self.device.thread_running:
                # Wait until the main DeviceThread enables scripts and there are
                # un-started scripts available in the list.
                while not self.device.scripts_enabled or \
                      self.device.scripts_started_idx >= len(self.device.scripts):
                    self.condition.wait()
                    if not self.device.thread_running: # Check for shutdown signal
                        return

                # Manually "pop" a script from the list-based queue.
                script_tuple = self.device.scripts[self.device.scripts_started_idx]
                self.device.scripts_started_idx += 1
                self.condition.notify_all()

                # Release the main lock to run the script, allowing other
                # workers to pick up other scripts in parallel.
                self.lock.release()
                self.run_script(script_tuple[0], script_tuple[1])
                self.lock.acquire() # Re-acquire lock to update shared state.

                # Atomically increment the done counter and notify the main thread.
                self.device.scripts_done_idx += 1
                self.done_condition.notify_all()

    def run_script(self, script, location):
        """Executes a single script, handling complex distributed locking.

        This method coordinates with other devices to ensure that a computation
        at a specific `location` only happens when all required devices are free.
        """
        # Use the location-specific Condition for fine-grained locking.
        with self.device.location_conditions[location]:
            # Identify all devices (self and neighbors) needed for the script.
            script_devices = []
            for device in self.device.neighbours:
                if device.has_data(location):
                    script_devices.append(device)
            if self.device.has_data(location):
                script_devices.append(self.device)

            if not script_devices:
                return

            # This loop implements a distributed lock. It waits until all
            # devices needed for this computation are no longer busy.
            while True:
                free = all(not device.is_busy(location) for device in script_devices)
                if free:
                    break
                # Wait to be notified by another thread that a device is free.
                self.device.location_conditions[location].wait()

            # "Acquire" the distributed lock by marking all devices as busy.
            for device in script_devices:
                device.set_busy(location, True)
            self.device.location_conditions[location].notify_all()

            # Data gathering can now proceed.
            script_data = [dev.get_data(location) for dev in script_devices]
            
            # The script execution itself can happen outside the lock.
            result = script.run(script_data)

            # Re-acquire lock to write results and release the busy flags.
            for device in script_devices:
                device.set_data(location, result)
                device.set_busy(location, False) # "Release" the distributed lock.
            
            # Notify any other workers waiting on this location that it's now free.
            self.device.location_conditions[location].notify_all()


class DeviceThread(Thread):
    """The main supervisor thread for a single Device."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main lifecycle loop for the device.
        
        Orchestrates the device's participation in synchronized time steps.
        """
        # Wait for the master to finish initial setup.
        self.device.ready_to_start.wait()

        while True:
            # 1. Synchronize with all other devices at the start of a time step.
            self.device.timestep_barrier.wait()

            # Get neighbors for this time step.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                break # Exit condition for the simulation.

            # 2. Enable and signal the worker pool to start processing scripts.
            with self.device.scripts_lock:
                self.device.scripts_started_idx = 0
                self.device.scripts_done_idx = 0
                self.device.scripts_enabled = True
                self.device.scripts_condition.notify_all()

            # 3. Wait until all scripts for this step have been assigned and completed.
            with self.device.scripts_lock:
                while not self.device.scripts_assigned or \
                      self.device.scripts_done_idx < len(self.device.scripts):
                    self.device.scripts_done_condition.wait()
                self.device.scripts_enabled = False

        # 4. Signal all worker threads to shut down.
        with self.device.scripts_lock:
            self.device.thread_running = False
            self.device.scripts_condition.notify_all()
