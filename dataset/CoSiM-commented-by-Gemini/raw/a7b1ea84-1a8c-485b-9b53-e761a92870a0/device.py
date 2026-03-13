"""
This module defines a complex master-worker simulation for a device in a
distributed sensor network.

The architecture uses a `DeviceMaster` thread to coordinate the simulation
timepoint by timepoint, and a pool of `DeviceWorker` threads to execute
script tasks. A `Queue` is used as a task buffer between the master and
workers. The logic is notable for its two-phase task processing within a
single timepoint: it first triggers a re-computation of all existing scripts
and then processes newly assigned scripts.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue

class Device(object):
    """
    Represents a single device, holding sensor data and managing a master-worker
    threading architecture for script processing.
    """

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
        self.timepoint_done = Event()  # Signals that script assignment for a timepoint is complete.

        # --- Master-Worker Communication & State ---
        self.buffer = Queue()  # Task queue for workers.
        self.fresh = []  # A temporary list for newly assigned scripts.
        self.scripts_by_location = {} # Stores all assigned scripts, organized by location.

        # --- Threads ---
        self.master = DeviceMaster(self)
        self.master.start()
        self.workers = [DeviceWorker(self) for _ in xrange(8)]
        for worker in self.workers:
            worker.start()

        # --- Synchronization ---
        # A dictionary of locks to protect this device's local sensor data.
        self.local_lock = {loc: Lock() for loc in self.sensor_data.keys()}
        self.barrier = None # The global timepoint synchronization barrier.
        self.location_lock = None # Note: This seems unused in the provided code.

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared barrier for all devices.
        
        This is called by device 0 to create and distribute the barrier.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for i in xrange(1, len(devices)):
                devices[i].barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a new script. A `None` script signals the end of assignment.
        """
        if script is not None:
            # New scripts are temporarily held in the 'fresh' list.
            self.fresh.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Acquires a local lock and returns data for a given location.

        NOTE: This lock is released in `set_data`. This couples the two methods
        and means the lock is held for the duration of a script's execution.
        """
        data = None
        if location in self.sensor_data:
            self.local_lock[location].acquire()
            data = self.sensor_data[location]
        return data

    def set_data(self, location, data):
        """Updates data for a given location and releases the local lock."""
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.local_lock[location].release()

    def shutdown(self):
        """Joins all master and worker threads to shut down the device."""
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceWorker(Thread):
    """
    A worker thread that executes script tasks from the device's shared buffer.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Worker %d" % device.device_id)
        self.device = device
        self.master = self.device.master

    def _run_one_script(self, script, location):
        """Helper method to execute a single script."""
        script_data = []
        # Aggregate data from neighbors.
        for device in self.master.neighbours:
            if device != self.device:
                data = device.get_data(location) # Acquires lock on other device.
                if data is not None:
                    script_data.append(data)

        # Aggregate data from the local device.
        data = self.device.get_data(location) # Acquires lock on this device.
        if data is not None:
            script_data.append(data)

        # Run script and propagate result.
        if script_data:
            result = script.run(script_data)
            for device in self.master.neighbours:
                if device != self.device:
                    device.set_data(location, result) # Releases lock on other device.
            self.device.set_data(location, result) # Releases lock on this device.

    def _run_all_by_location(self, location):
        """Helper method to execute ALL scripts associated with a location."""
        for script in self.master.scripts_by_location[location]:
            self._run_one_script(script, location)

    def run(self):
        """The main loop for a worker thread, processing tasks from the buffer."""
        while True:
            (script, location) = self.master.buffer.get()

            # A (None, None) task is the poison pill to terminate the thread.
            if location is None:
                self.master.buffer.task_done()
                break
            
            # A (None, loc) task is a trigger to re-run all scripts for that location.
            if script is None:
                self._run_all_by_location(location)
            else:
                # Otherwise, it's a new script to be run once.
                self._run_one_script(script, location)

            self.master.buffer.task_done()


class DeviceMaster(Thread):
    """
    The master thread that coordinates the device's state across a timepoint.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Master %d" % device.device_id)
        self.device = device
        self.neighbours = None
        # Create convenient aliases for shared state.
        self.buffer = self.device.buffer
        self.fresh = self.device.fresh
        self.scripts_by_location = self.device.scripts_by_location

    def run(self):
        """The main control loop for the master thread."""
        while True:
            self.neighbours = self.device.supervisor.get_neighbours()

            # `None` neighbors signal the end of the simulation.
            if self.neighbours is None:
                # Put poison pills in the queue for all worker threads.
                for _ in self.device.workers:
                    self.buffer.put((None, None))
                break

            # --- Phase 1: Trigger re-computation of existing scripts ---
            # For every location we know about, enqueue a task to re-run all its scripts.
            for loc in self.scripts_by_location.keys():
                self.buffer.put((None, loc))

            # --- Phase 2: Wait for and process new scripts ---
            # Wait for the supervisor to signal that new script assignment is done.
            self.device.timepoint_done.wait()

            # Process all newly assigned scripts from the 'fresh' list.
            while self.fresh:
                elem = (script, location) = self.fresh.pop(0)

                # Add the new script to the persistent store.
                if location not in self.scripts_by_location:
                    self.scripts_by_location[location] = [script]
                else:
                    self.scripts_by_location[location].append(script)
                
                # Enqueue the new script for immediate execution.
                self.buffer.put(elem)

            # --- Synchronization ---
            # Wait for workers to finish all tasks (both re-computations and new scripts).
            self.buffer.join()
            # Wait at the global barrier for all other devices to finish their timepoint.
            self.device.barrier.wait()
            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()
