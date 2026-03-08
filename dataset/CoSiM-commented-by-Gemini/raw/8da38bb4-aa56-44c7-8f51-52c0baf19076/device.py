# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices for a sensor network simulation.

This script defines a simulation framework where devices, each running in their own
main thread, process data in discrete, synchronized time steps. For each time step,
a device spawns a temporary pool of worker threads to execute assigned scripts in parallel.
"""

from threading import Event, Lock, Thread, RLock, Semaphore


class ReusableBarrier():
    """A reusable barrier implementation for synchronizing multiple threads.

    This barrier uses a two-phase protocol to allow it to be used multiple times.
    Threads wait at the barrier until all participating threads have arrived,
    after which they are all released.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

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
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of the barrier.
                count_threads[0] = self.num_threads
        # All threads wait here until the semaphore is released.
        threads_sem.acquire()


class Device(object):
    """Represents a single device in the distributed network simulation.

    Each device has one main control thread (`DeviceThread`) and manages its own
    state and data.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's local sensor readings.
        supervisor (object): A reference to the central supervisor.
        scripts (list): A list of scripts to be executed in the current time step.
        timepoint_done (Event): An event to signal that script assignment is complete.
        thread (DeviceThread): The main control thread for this device.
        lock (RLock): A re-entrant lock to protect access to `sensor_data`.
        script_lock (RLock): A re-entrant lock to protect access to the `scripts` list.
        barrier (ReusableBarrier): A shared barrier to synchronize with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device, creating and starting its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.lock = RLock()
        self.script_lock = RLock()
        self.run_lock = RLock() # Note: This lock is initialized but never used.

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up a shared barrier for a list of devices.

        Device 0 is arbitrarily chosen to create the barrier, which is then
        distributed to all other devices for global time step synchronization.
        """
        if self.device_id is 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to the device, called by the supervisor."""
        with self.script_lock:
            if script is not None:
                self.scripts.append((script, location))
                self.script_received.set()
            else:
                # A None script is a sentinel indicating the end of a time step's assignments.
                self.timepoint_done.set()

    def get_data(self, location):
        """Thread-safely retrieves sensor data for a given location."""
        with self.lock:
            result = self.sensor_data.get(location)
        return result

    def set_data(self, location, data):
        """Thread-safely updates sensor data for a given location."""
        with self.lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a single Device.

    This thread orchestrates the work for a device within each simulation time step.
    It fetches neighbors, waits for scripts, and then spawns temporary worker
    threads to execute the scripts.
    """

    def __init__(self, device):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop, executed in discrete time steps."""
        while True:
            # At the start of a step, get the current set of neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbors is the signal to shut down.
                break

            # The device includes itself in the list for data processing.
            neighbours.append(self.device)

            # Wait until the supervisor signals that all scripts for this step are assigned.
            self.device.timepoint_done.wait()

            # --- Dynamic Thread Pool for Concurrent Work ---
            # For each time step, spawn a new pool of worker threads.
            num_threads = 8
            threads = [Thread(target=self.concurrent_work,
                              args=(neighbours, i, num_threads)) for i in range(num_threads)]

            for thread in threads:
                thread.start()

            # Wait for all worker threads for this time step to complete.
            for thread in threads:
                thread.join()

            # --- Global Synchronization Point (End of Timepoint) ---
            # Wait at the shared barrier. No device will proceed to the next time step
            # until all devices have completed the current one.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()

    def concurrent_work(self, neighbours, thread_id, num_threads):
        """The work function for the dynamically spawned worker threads.

        Args:
            neighbours (list): List of neighboring devices (including self).
            thread_id (int): The ID of this worker thread (0 to num_threads-1).
            num_threads (int): The total number of worker threads for this device.
        """
        # Each thread processes a statically assigned subset of the scripts.
        for (script, location) in self.keep_assigned(self.device.scripts, thread_id, num_threads):
            script_data = []

            # Aggregate data from all neighbors for the target location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            if script_data:
                # Execute the script on the aggregated data.
                result = script.run(script_data)

                # --- Data Dissemination (Potential Race Condition) ---
                # Update the data on all neighbors. This implements a max-consensus algorithm.
                # WARNING: This block is not atomic across different devices. Two threads from
                # different devices could read the same `device.get_data`, compute a new `max`,
                # and then write back, causing one update to be lost.
                for device in neighbours:
                    res = max(result, device.get_data(location))
                    device.set_data(location, res)

    def keep_assigned(self, scripts, thread_id, num_threads):
        """Partitions the script list among the worker threads.

        This uses a simple modulo-based static partitioning scheme.

        Returns:
            list: A list of scripts assigned to this specific thread.
        """
        assigned_scripts = []
        for i, script in enumerate(scripts):
            if i % num_threads is thread_id:
                assigned_scripts.append(script)

        return assigned_scripts
