# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices for a sensor network simulation.

This script defines the core components for simulating a network of devices that
process data based on scripts assigned by a central supervisor. It uses advanced
concurrency primitives to manage the state and interactions between devices and
their internal worker threads.
"""

from threading import Event, Thread, Lock, Semaphore
from collections import deque


class Device(object):
    """Represents a single device in the distributed network simulation.

    Each device manages its own sensor data, a pool of worker threads, and a set of
    synchronization objects to coordinate with other devices and a central supervisor.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's local sensor readings.
        supervisor (object): A reference to the central supervisor managing the simulation.
        scripts (list): A list to store scripts assigned by the supervisor for a timepoint.
        thread (list): A pool of `DeviceThread` worker instances.
        local_lock (Lock): A mutex to protect the device's shared internal state.
        zones (dict): A shared dictionary mapping data locations to locks.
        num_threads (int): The number of worker threads for this device.
        neighbours (list): A list of neighboring `Device` objects.
        zones_lock (Lock): A shared mutex for safely accessing the `zones` dictionary.
        local_barrier (ReusableBarrier): A barrier to synchronize the device's own threads.
        global_barrier (ReusableBarrier): A barrier to synchronize all threads across all devices.
        todo_scripts (deque): A queue of scripts to be executed by the worker threads.
        script_received (Event): Signals that a new script has been assigned.
        timepoint_done (Event): Signals that all scripts for a timepoint have been assigned.
        got_neighbours (Event): Used to coordinate fetching of neighbor information.
        got_scripts (Event): Used to coordinate the collection of assigned scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = []

        self.local_lock = Lock()

        self.zones = None
        self.num_threads = 8
        self.got_neighbours = Event()

        self.got_scripts = Event()
        self.neighbours = []

        self.zones_lock = None

        self.local_barrier = ReusableBarrier(self.num_threads)

        self.global_barrier = ReusableBarrier(1)
        self.todo_scripts = None

        # Create and start the pool of worker threads.
        for _ in range(self.num_threads):
            self.thread.append(DeviceThread(self))

        for i in range(self.num_threads):
            self.thread[i].start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources for a list of devices.

        This method acts like a static initializer, providing all devices with
        references to shared state, including a global barrier and shared locks.

        Args:
            devices (list): A list of `Device` objects in the simulation.
        """
        zones = {}

        # The global barrier must synchronize all threads from all devices.
        global_barrier = ReusableBarrier(devices[0].num_threads * len(devices))

        zones_lock = Lock()
        for dev in devices:
            dev.zones = zones
            dev.global_barrier = global_barrier
            dev.zones_lock = zones_lock

    def assign_script(self, script, location):
        """Assigns a script to the device for a specific location.

        This method is called by the supervisor. A `None` script is a sentinel
        to indicate that all scripts for the current timepoint have been sent.

        Args:
            script (object): The script to be executed.
            location (any): The data location the script applies to.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Signal that the script assignment phase for this timepoint is over.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its worker threads."""
        for thread in self.thread:
            thread.join()


class DeviceThread(Thread):
    """A worker thread for a Device.

    Executes the main simulation loop, processing scripts and synchronizing
    with other threads and devices.
    """

    def __init__(self, device):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the device thread."""
        self.device.got_neighbours.set()
        self.device.got_scripts.set()
        while True:
            # --- Global Synchronization Point (Start of Timepoint) ---
            # All threads from all devices wait here, ensuring all start the
            # timepoint together.
            self.device.global_barrier.wait()

            # --- Neighbor Discovery Phase ---
            # Use a lock to ensure only one thread from this device fetches
            # the neighbor list from the supervisor.
            self.device.local_lock.acquire()
            if self.device.got_neighbours.isSet():
                self.device.timepoint_done.clear()
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.got_neighbours.clear()
            self.device.local_lock.release()

            # All local threads wait here to ensure they all have the updated neighbor list.
            self.device.local_barrier.wait()
            self.device.got_neighbours.set()

            # If neighbors is None, it's a signal to shut down.
            if self.device.neighbours is None:
                break

            # Wait until the supervisor signals that all scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()

            # --- Script Collection Phase ---
            # One thread gathers all assigned scripts into a shared deque.
            self.device.local_lock.acquire()
            if self.device.got_scripts.isSet():
                self.device.todo_scripts = deque(self.device.scripts)
                self.device.got_scripts.clear()
            self.device.local_lock.release()

            # All local threads sync up before starting script execution.
            self.device.local_barrier.wait()
            self.device.got_scripts.set()

            # --- Script Execution Phase ---
            while True:
                self.device.local_lock.acquire()

                # If the script queue is empty, this thread is done for this timepoint.
                if not self.device.todo_scripts:
                    self.device.local_lock.release()
                    break

                # Pop a script from the shared queue atomically.
                (script, location) = self.device.todo_scripts.popleft()

                # --- Zone Locking ---
                # Ensure a lock exists for the script's target location.
                self.device.zones_lock.acquire()
                if location not in self.device.zones:
                    self.device.zones[location] = Lock()
                self.device.zones_lock.release()

                # Acquire the lock for this specific location. This prevents
                # other devices from working on the same data point concurrently.
                self.device.zones[location].acquire()
                self.device.local_lock.release()

                # --- Data Aggregation ---
                script_data = []
                # Gather data from all neighbors for the target location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather this device's own data.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # --- Script Execution and Data Dissemination ---
                if script_data:
                    # Run the script on the aggregated data.
                    result = script.run(script_data)

                    # Broadcast the result to all neighbors and update local data.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the lock for the location, allowing other devices to access it.
                self.device.zones[location].release()


class ReusableBarrier():
    """A reusable barrier implementation for synchronizing multiple threads.

    This barrier uses a two-phase protocol to allow it to be used multiple times.
    Threads wait at the barrier until all participating threads have arrived,
    after which they are all released.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Use a list for the counter to make it mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase.

    def wait(self):
        """Causes a thread to wait at the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of the barrier.
                count_threads[0] = self.num_threads
        # All threads wait here until the semaphore is released.
        threads_sem.acquire()
