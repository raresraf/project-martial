"""
This module defines a device simulation framework where each device, at each
time step, creates a new dynamic pool of worker threads to process scripts
from a queue.

The architecture consists of three primary classes:
- Device: Represents a node in the network. It is controlled by a master
  `DeviceThread`.
- DeviceThread: The master thread for a device. It orchestrates the time steps
  by creating, running, and destroying a pool of `Workerr` threads in each step.
- Workerr: A worker thread that consumes script-execution tasks from a queue.
"""

from threading import Event, Thread, Lock
from Queue import Queue
import barrier # Assumed to be a module providing ReusableBarrierCond


class Device(object):
    """
    Represents a single device in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        
        # --- Synchronization Objects ---
        self.timepoint_done = Event()
        self.everyone = [] # A list of all devices in the simulation.
        self.barrier = None # The global time step barrier.
        # A fixed-size list of locks for location-based synchronization.
        self.locations_lock = [None] * 100
        
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global synchronization objects for all devices.

        The root device (id 0) creates and distributes the shared barrier,
        a list of all devices, and the list of location locks.
        """
        if self.device_id == 0:
            self.barrier = barrier.ReusableBarrierCond(len(devices))
            for device in devices:
                device.barrier = self.barrier

            self.everyone = devices
            for device in devices:
                device.everyone = self.everyone

            for i in range(100):
                self.locations_lock[i] = Lock()
            for device in devices:
                device.locations_lock = self.locations_lock

    def assign_script(self, script, location):
        """Assigns a script to be processed in the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data from a location (not intrinsically thread-safe)."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a location (not intrinsically thread-safe)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It dynamically creates a new worker
    pool and task queue for each time step.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []
        self.jobs = Queue(maxsize=0)
        self.exit = False
        self.time_done = False
        self.neighbours = []

    def run(self):
        """The main lifecycle loop of the device."""
        while True:
            self.time_done = False
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                self.time_done = True
                self.exit = True

            if not self.exit:
                self.device.timepoint_done.wait()

            if not self.time_done:
                # --- Dynamic Worker Pool Creation ---
                # Create a new pool of 8 workers for this time step.
                self.workers = [Workerr(self, self.neighbours) for _ in range(8)]
                for worker in self.workers:
                    worker.start()

                # Populate the queue with this time step's jobs.
                for script in self.device.scripts:
                    self.jobs.put(script)

                # Wait for workers to process all jobs.
                self.jobs.join()

                # --- Worker Pool Shutdown ---
                # Send a termination signal to each worker.
                for _ in self.workers:
                    self.jobs.put(None)
                self.jobs.join() # Wait for sentinels to be processed.

                # Wait for all worker threads to terminate.
                for worker in self.workers:
                    worker.join()
                self.workers = []
                self.time_done = True

            # --- Global Synchronization and Cleanup ---
            if self.time_done:
                self.device.timepoint_done.clear()
                # Access the barrier through the shared 'everyone' list.
                self.device.everyone[0].barrier.wait()
                if self.exit:
                    break


class Workerr(Thread):
    """A worker thread that consumes tasks from its parent's job queue."""
    def __init__(self, parent_thread, neighbours):
        Thread.__init__(self)
        self.parent_thread = parent_thread # This is the DeviceThread instance.
        self.neighbours = neighbours

    def run(self):
        """Continuously gets and processes jobs from the queue."""
        while True:
            job_to_do = self.parent_thread.jobs.get()
            if job_to_do is None:
                self.parent_thread.jobs.task_done()
                break # Terminate signal received.

            (script, location) = job_to_do
            # Use the global lock for this location to ensure serial access.
            with self.parent_thread.device.locations_lock[location]:
                script_data = []
                # Gather data.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.parent_thread.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute script and propagate results.
                if script_data:
                    result = script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(location, result)
                    self.parent_thread.device.set_data(location, result)

            self.parent_thread.jobs.task_done()
