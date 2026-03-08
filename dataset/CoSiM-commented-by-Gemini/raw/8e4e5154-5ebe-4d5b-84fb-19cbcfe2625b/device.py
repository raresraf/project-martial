# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices using a producer-consumer pattern.

This script implements a simulation where each device has a main control thread
acting as a "producer" and a persistent pool of "consumer" worker threads. The
producer thread receives scripts and places them on a work queue, and the workers
consume from this queue to execute the scripts in parallel.
"""

from threading import Event, Thread, Lock, Semaphore, Condition
from Queue import Queue
# The ReusableBarrierSem class is defined at the end of the file, so this import
# is redundant if the file is self-contained.
from barrier import ReusableBarrierSem


class Device(object):
    """Represents a single device, managing its state and a pool of worker threads.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): The device's local sensor data.
        supervisor (object): A reference to the central supervisor.
        all_devices (list): A reference to all devices in the simulation.
        event_timepoint_done (Event): Signals that script assignment is complete.
        event_setup_done (Event): Signals that the initial setup is complete.
        event_stop_threads (Event): Signals worker threads to terminate.
        lock_data (Lock): A lock to protect writes to this device's `sensor_data`.
        lock_locations (list): A globally shared list of locks for each data location.
        queue_scripts (Queue): A work queue for scripts to be processed by workers.
        semaphore_queue (Semaphore): A semaphore to signal available work in the queue.
        barrier_devices (ReusableBarrierSem): A barrier for global time step synchronization.
        condition_variable (Condition): Used to signal when the work queue is empty.
        thread (DeviceThread): The main producer/control thread for this device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device and its associated control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.all_devices = []

        # --- Synchronization Primitives ---
        self.event_timepoint_done = Event()
        self.event_setup_done = Event()
        self.event_stop_threads = Event()
        self.lock_data = Lock()
        self.lock_locations = []
        self.queue_scripts = Queue()
        self.semaphore_queue = Semaphore(0)
        self.barrier_devices = None
        self.condition_variable = Condition(Lock())

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources (barrier, locks) for all devices."""
        self.all_devices = devices
        # Device 0 acts as the leader for setup.
        if self.device_id == 0:
            nr_of_devices = len(devices)
            barrier_devices = ReusableBarrierSem(nr_of_devices)

            # Pre-allocate a global lock for each possible location (hardcoded to 24).
            for _ in range(24):
                self.lock_locations.append(Lock())

            # Distribute shared resources to all devices.
            for device in devices:
                device.barrier_devices = barrier_devices
                device.lock_locations = self.lock_locations
                device.event_setup_done.set()

    def assign_script(self, script, location):
        """Assigns a script, called by the supervisor."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of assignments for the time step.
            self.event_timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. Not internally synchronized on read."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Thread-safely updates sensor data."""
        with self.lock_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main producer/control thread for a Device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main simulation loop: produces work for consumer threads."""
        # Wait until the initial device setup is complete.
        self.device.event_setup_done.wait()

        # Create and start a persistent pool of worker threads.
        worker_threads = []
        for thread_id in range(8):
            thread = WorkerThread(self.device, thread_id)
            worker_threads.append(thread)
            worker_threads[-1].start()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Shutdown signal
                break

            # Wait for the supervisor to finish assigning scripts.
            self.device.event_timepoint_done.wait()

            # --- Producer Logic ---
            # Place all assigned scripts onto the work queue.
            for (script, location) in self.device.scripts:
                thread_script = (script, location, neighbours)
                self.device.queue_scripts.put(thread_script)
                # Signal a worker that a job is available.
                self.device.semaphore_queue.release()
            
            self.device.scripts = [] # Clear scripts for the next timepoint
            self.device.event_timepoint_done.clear()

            # Wait until the worker threads have emptied the queue.
            with self.device.condition_variable:
                while not self.device.queue_scripts.empty():
                    self.device.condition_variable.wait()

            # Synchronize with all other devices before starting the next time step.
            self.device.barrier_devices.wait()

        # --- Shutdown Logic ---
        self.device.event_stop_threads.set()
        # Unblock any workers waiting on the semaphore.
        for _ in range(8):
            self.device.semaphore_queue.release()
        for thread in worker_threads:
            thread.join()


class WorkerThread(Thread):
    """A consumer thread that processes scripts from a work queue."""
    def __init__(self, device, my_id):
        Thread.__init__(self, name="Worker Thread %d" % my_id)
        self.device = device

    def run(self):
        """Main consumer loop."""
        while True:
            # Wait for a job to be added to the queue.
            self.device.semaphore_queue.acquire()

            if self.device.event_stop_threads.is_set():
                break

            # Get a script package from the queue.
            (script, location, neighbours) = self.device.queue_scripts.get()

            # Acquire the global lock for this location to ensure exclusive access.
            with self.device.lock_locations[location]:
                script_data = []
                # Aggregate data from self and neighbors.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Execute the script and disseminate the results.
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

            # Notify the producer that one item has been finished.
            with self.device.condition_variable:
                self.device.condition_variable.notify()


# This class is defined here to make the file self-contained.
class ReusableBarrierSem():
    """A reusable barrier for thread synchronization using semaphores."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        self.phase1()
        self.phase2()

    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
