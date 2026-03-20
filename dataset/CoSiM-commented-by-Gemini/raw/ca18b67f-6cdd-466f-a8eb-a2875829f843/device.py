"""
This module provides a framework for simulating a distributed network of devices.

Each device consists of a master `DeviceThread` that manages a pool of 8 `DeviceWorker`
threads. Scripts are distributed among workers based on location affinity.
Synchronization is handled by a global `ReusableBarrier` and a global lock for
accessing neighbors, which may act as a bottleneck. The implementation appears
to be for Python 2.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue

class ReusableBarrier(object):
    """A reusable, two-phase barrier implemented with semaphores.

    This implementation uses a list-based counter as a workaround for Python 2's
    lack of mutable integers in outer scopes.
    """

    def __init__(self, num_threads):
        """Initializes the barrier."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """Represents a single device in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.result_queue = Queue.Queue()  # Note: This queue is unused.
        self.set_lock = Lock()
        self.neighbours_lock = None
        self.neighbours_barrier = None
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up and distributes shared synchronization objects.

        The first device in the list acts as a master, creating a global lock
        and barrier for all other devices.
        """
        if self.device_id == devices[0].device_id:
            self.neighbours_lock = Lock()
            self.neighbours_barrier = ReusableBarrier(len(devices))
        else:
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to the device for execution."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location.
        
        Warning: This method is not thread-safe and can lead to race conditions
        when called concurrently from multiple workers.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Thread-safely updates sensor data for a given location."""
        with self.set_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The master thread for a device, managing a pool of workers."""

    def __init__(self, device):
        """Initializes the master thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []

    def run(self):
        """The main execution loop for the master thread."""
        while True:
            # This global lock is a potential performance bottleneck.
            with self.device.neighbours_lock:
                neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break  # End of simulation

            self.device.script_received.wait()

            # Create a new pool of workers for each time step.
            self.workers = [DeviceWorker(self.device, i, neighbours) for i in range(8)]

            # Distribute scripts to workers based on a load-balancing strategy.
            for (script, location) in self.device.scripts:
                added = False
                for worker in self.workers:
                    if location in worker.locations:
                        worker.add_script(script, location)
                        added = True
                
                if not added:
                    # Assign to the worker with the fewest locations.
                    chosen_worker = min(self.workers, key=lambda w: len(w.locations))
                    chosen_worker.add_script(script, location)

            for worker in self.workers:
                worker.start()

            for worker in self.workers:
                worker.join()

            # Synchronize with other devices before the next time step.
            self.device.neighbours_barrier.wait()
            self.device.script_received.clear()


class DeviceWorker(Thread):
    """A worker thread that executes a batch of scripts."""

    def __init__(self, device, worker_id, neighbours):
        """Initializes a worker thread."""
        Thread.__init__(self)
        self.device = device
        self.worker_id = worker_id
        self.scripts = []
        self.locations = []
        self.neighbours = neighbours

    def add_script(self, script, location):
        """Adds a script to this worker's execution batch."""
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        """Executes all scripts assigned to this worker.
        
        Warning: Data access in this method is not properly synchronized,
        which can lead to race conditions. `get_data` is called without a lock.
        """
        for (script, location) in zip(self.scripts, self.locations):
            script_data = []
            
            # Gather data from neighbors and self.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Run script and distribute results.
                res = script.run(script_data)
                for device in self.neighbours:
                    device.set_data(location, res)
                self.device.set_data(location, res)

    def run(self):
        """The thread's entry point."""
        self.run_scripts()
