# -*- coding: utf-8 -*-
"""
This module simulates a network of devices using a thread-per-task concurrency
model. Each device's main thread spawns a new worker thread for every assigned
script. Synchronization between devices is managed by a custom semaphore-based
reusable barrier, and concurrency on data locations is handled by a lazily-
populated dictionary of locks.

Classes:
    ReusableBarrier: A custom semaphore-based reusable barrier.
    Device: Represents a node in the network.
    DeviceThread: The main control loop and thread manager for a device.
"""

from threading import Lock, Semaphore, Thread, Event


class ReusableBarrier(object):
    """A standard two-phase reusable barrier implemented with semaphores."""
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # A list is used to make the counter mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes the calling thread to wait at the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """Represents a single device in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # These are placeholders; they are replaced in setup_devices.
        self.barrier = "barrier"
        self.dictionary = "dict"

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources.
        
        Note: This setup method is not leader-based and is therefore susceptible
        to race conditions. The resources created by the last device to call
        this method will be used by all.
        """
        self.barrier = ReusableBarrier(len(devices))
        self.dictionary = {}  # Shared dictionary for location locks.

        for device in devices:
            device.barrier = self.barrier
            device.dictionary = self.dictionary

    def assign_script(self, script, location):
        """
        Assigns a script and lazily creates a lock for its location if one
        does not already exist.

        Note: The check-then-create logic for the lock is not atomic and
        can lead to a race condition if multiple devices are assigned scripts
        for the same new location concurrently.
        """
        if location not in self.dictionary:
            self.dictionary[location] = Lock()

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Not used in this implementation.
        else:
            # Signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control loop for a device, spawning worker threads for each task."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def work(self, script, location, neighbours):
        """
        The target function for worker threads. It processes one script.
        
        It acquires a lock for the specified location, gathers data from itself
        and neighbours, runs the script, and disseminates the result.
        """
        script_data = []
        
        # Acquire a per-location lock to ensure data consistency.
        self.device.dictionary[location].acquire()
        try:
            # --- Data Gathering Phase ---
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # --- Execution and Dissemination Phase ---
            if script_data:
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
        finally:
            # Always release the lock.
            self.device.dictionary[location].release()

    def run(self):
        """
        The main simulation loop.
        
        Synchronization Logic:
        1. Wait for the supervisor to assign all scripts.
        2. Spawn and start a new thread for each script.
        3. Wait for all spawned threads to complete.
        4. Wait at a global barrier to synchronize with all other devices.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Supervisor signals shutdown.
                break

            # Wait for the signal that scripts are assigned.
            self.device.timepoint_done.wait()
            # This set() is likely a bug, as it negates the wait().
            self.device.timepoint_done.set()

            # --- Thread-per-task Execution ---
            worker_threads = {}
            for (script, location) in self.device.scripts:
                thread = Thread(name="worker",
                                target=self.work,
                                args=(script, location, neighbours))
                worker_threads[id(script)] = thread
                thread.start()

            # Wait for all worker threads to complete their execution.
            for thread_id in worker_threads:
                worker_threads[thread_id].join()

            # --- Global Synchronization Point ---
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
            self.device.scripts = []