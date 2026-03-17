"""
This module simulates a distributed device network and appears to be a concatenation of several
related classes: Device, DeviceThread, WorkPool, and ScriptExecutor. It uses a thread pool
pattern (`WorkPool`) to execute scripts, with a main control thread (`DeviceThread`) on each
device managing the simulation's timepoints and script assignments.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue
# The following imports assume the presence of corresponding files (barrier.py, workPool.py, scriptexecutor.py)
# in the execution environment, though the classes are defined within this same file.
from barrier import ReusableBarrierSem
from workPool import WorkPool
from scriptexecutor import ScriptExecutor


class Device(object):
    """
    The central class representing a device. It coordinates a main control thread, a pool
    of worker threads for script execution, and manages shared synchronization objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device, its work pool, and starts its main control thread.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): The local data store for this device.
            supervisor (object): The central supervisor managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.workpool = WorkPool(4, self)

        self.scripts = []
        # This appears to be a temporary storage to hold scripts for re-execution in the next timepoint.
        self.script_storage = []
        self.locks = []
        self.barrier = None
        self.neighbours = None

        self.script_lock = Lock()
        # Event to signal interaction from the supervisor (e.g., script assignment).
        self.supervisor_interact = Event()
        # Event to signal that the script assignment phase for a timepoint is over.
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        A centralized setup routine run by the device with the minimum ID. It creates
        and distributes a shared barrier and a set of per-location locks to all devices.
        """
        ids = [device.device_id for device in devices]
        loc = [location for device in devices for location in device.sensor_data]

        max_locations = max(loc) + 1 if loc else 0
        # Pre-condition: Only the device with the lowest ID performs the setup.
        if self.device_id == min(ids):
            barrier = ReusableBarrierSem(len(ids))
            locks = [Lock() for _ in range(max_locations)]
            # Invariant: Distributes the same shared objects to all devices.
            for device in devices:
                device.assign_barrier(barrier)
                device.set_locks(locks)

    def assign_barrier(self, barrier):
        """Assigns the shared barrier object to this device."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """

        Assigns a script to the device. Called by the supervisor. A `None` script
        signals the end of the assignment phase for the current timepoint.
        """
        if script is not None:
            self.script_lock.acquire()
            self.scripts.append((script, location))
            self.script_lock.release()
        else:
            # Signals the control thread that the timepoint is complete.
            self.timepoint_done.set()
        # Signals the control thread that an interaction has occurred.
        self.supervisor_interact.set()

    def get_data(self, location):
        """Retrieves data from the local sensor store for a given location."""
        return self.sensor_data.get(location)

    def wait_on_scripts(self):
        """Blocks until the work pool has completed all enqueued scripts."""
        self.workpool.wait()

    def set_data(self, location, data):
        """Updates data in the local sensor store for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to finish."""
        self.thread.join()

    def set_locks(self, locks):
        """Assigns the list of shared, per-location lock objects to this device."""
        self.locks = locks

    def lock(self, location):
        """Acquires the specific lock for a given data location."""
        self.locks[location].acquire()

    def unlock(self, location):
        """Releases the specific lock for a given data location."""
        self.locks[location].release()

    def execute_scripts(self):
        """
        Moves scripts from the device's script list to the work pool for execution.
        """
        self.script_lock.acquire()
        for (script, location) in self.scripts:
            self.script_storage.append((script, location))
            self.workpool.add_data(script, location)
        del self.scripts[:]
        self.script_lock.release()


class DeviceThread(Thread):
    """
    The main control thread for a device, managing the lifecycle of a simulation timepoint
    in an event-driven manner.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop, organized into discrete timepoints."""
        while True:
            # Fetches neighbors at the start of each major loop.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                # Termination signal from the supervisor.
                self.device.workpool.shutdown()
                return

            self.device.execute_scripts()

            # Inner loop manages a single timepoint, reacting to supervisor events.
            while True:
                # Waits for a signal from the supervisor (e.g., a new script).
                self.device.supervisor_interact.wait()
                self.device.supervisor_interact.clear()

                # Block Logic: If new scripts have been added, execute them.
                self.device.script_lock.acquire()
                if len(self.device.scripts) > 0:
                    self.device.script_lock.release()
                    self.device.execute_scripts()
                else:
                    self.device.script_lock.release()

                # Block Logic: Checks if the supervisor has signaled the end of the timepoint.
                if self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
                    
                    if len(self.device.scripts) > 0:
                        self.device.execute_scripts()

                    # Synchronization Point 1: Wait for all scripts in the workpool to finish.
                    self.device.wait_on_scripts()

                    # Synchronization Point 2: Wait for all other devices to reach this barrier.
                    self.device.barrier.wait()

                    # The executed scripts are restored, suggesting they are re-executed each timepoint.
                    self.device.scripts = self.device.script_storage
                    self.device.script_storage = []
                    break # Exit the inner loop to start the next timepoint.


class ScriptExecutor(Thread):
    """A worker thread that executes scripts from the WorkPool's queue."""

    def __init__(self, index, workpool, device):
        Thread.__init__(self, name="Worker Thread %d" % index)
        self.index = index
        self.workpool = workpool
        self.device = device

    def run(self):
        """Continuously fetches and executes tasks from the queue until shutdown."""
        while True:
            # Waits for data to be available in the queue.
            self.workpool.data.acquire()
            if self.workpool.done:
                return # Shutdown signal received.

            (script, location) = self.workpool.q.get()

            if self.device.neighbours is None:
                return # Check for shutdown signal again after getting from queue.

            # Synchronization Point: Acquire the global lock for this data location.
            self.device.lock(location)

            # Block Logic: Gather data from self and neighbors.
            script_data = []
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Execute script and propagate results.
            if script_data:
                result = script.run(script_data)
                for device in self.device.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)

            self.device.unlock(location)
            self.workpool.q.task_done()


class WorkPool(object):
    """A thread pool for executing scripts concurrently."""

    def __init__(self, num_threads, device):
        """
        Initializes the queue and starts a number of ScriptExecutor worker threads.
        Args:
            num_threads (int): The number of worker threads to create.
            device (Device): The parent device object.
        """
        self.device = device
        self.executors = []
        self.q = Queue()
        # This semaphore acts as a counter for items in the queue.
        self.data = Semaphore(0)
        self.done = False

        for i in range(num_threads):
            executor = ScriptExecutor(i, self, self.device)
            executor.start()
            self.executors.append(executor)

    def add_data(self, script, location):
        """Adds a script to the queue and signals a worker thread."""
        self.q.put((script, location))
        self.data.release()

    def wait(self):
        """Blocks until all items in the queue have been processed."""
        if not self.done:
            self.q.join()

    def shutdown(self):
        """Performs a graceful shutdown of all worker threads."""
        self.wait()
        self.done = True

        # Unblock any workers waiting on the semaphore so they can see the `done` flag.
        for _ in self.executors:
            self.data.release()

        for executor in self.executors:
            executor.join()