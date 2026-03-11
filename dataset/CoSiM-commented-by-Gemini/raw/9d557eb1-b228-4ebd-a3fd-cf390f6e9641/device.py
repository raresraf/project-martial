"""
This module simulates a network of devices processing sensor data in parallel.
This implementation features a two-phase reusable barrier and a dynamic worker
thread allocation strategy where new workers are created for each simulation step.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue

class ReusableBarrier(object):
    """
    A reusable two-phase barrier implemented with semaphores.

    This barrier ensures that all threads have reached a synchronization point
    before any of them are allowed to proceed. It uses two phases to prevent
    threads from one iteration from proceeding before all threads from the
    previous iteration have left the barrier.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Counter for the first phase.
        self.count_threads2 = [self.num_threads] # Counter for the second phase.
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase.

    def wait(self):
        """
        Blocks the calling thread until all threads have reached the barrier.
        Consists of two consecutive synchronization phases.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier synchronization.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire() # All threads wait here until released.

class Device(object):
    """
    Represents a single device in the simulated network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.result_queue = Queue.Queue() # This queue appears to be unused.
        self.set_lock = Lock() # Lock for thread-safe `set_data` operations.
        self.neighbours_lock = None # Shared lock for accessing neighbor list.
        self.neighbours_barrier = None # Shared barrier for step synchronization.

        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization primitives for a group of devices.
        One device acts as a leader to create the shared lock and barrier.
        """
        if self.device_id == devices[0].device_id:
            # This device is the leader, creating the shared objects.
            self.neighbours_lock = Lock()
            self.neighbours_barrier = ReusableBarrier(len(devices))
        else:
            # This is a follower device, using the leader's objects.
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier

        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be executed for a specific location."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of script assignment for the current step.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location in a thread-safe manner."""
        with self.set_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's control thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a Device. It dynamically creates worker threads
    for each simulation step to process assigned scripts.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []

    def run(self):
        """The main simulation loop for the device."""
        while True:
            # Acquire lock to safely get the list of neighbors.
            with self.device.neighbours_lock:
                neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # End of simulation.
                break

            # Wait for all scripts for the current step to be assigned.
            self.device.script_received.wait()

            # Create a new pool of worker threads for this step.
            # NOTE: Creating threads in a loop is generally inefficient.
            self.workers = [DeviceWorker(self.device, i, neighbours) for i in range(8)]

            # Distribute scripts to workers using a simple load-balancing scheme.
            for (script, location) in self.device.scripts:
                added = False
                # Prefer a worker already handling this location.
                for worker in self.workers:
                    if location in worker.locations:
                        worker.add_script(script, location)
                        added = True
                        break # Go to next script once assigned

                # If no worker handles this location, assign to the least loaded worker.
                if not added:
                    chosen_worker = min(self.workers, key=lambda w: len(w.locations))
                    chosen_worker.add_script(script, location)

            # Start all worker threads for this step.
            for worker in self.workers:
                worker.start()

            # Wait for all worker threads to complete their tasks.
            for worker in self.workers:
                worker.join()

            # Synchronize with all other devices before starting the next step.
            self.device.neighbours_barrier.wait()
            self.device.script_received.clear()

class DeviceWorker(Thread):
    """
    A worker thread that executes a batch of scripts for a single Device.
    """
    def __init__(self, device, worker_id, neighbours):
        Thread.__init__(self)
        self.device = device
        self.worker_id = worker_id
        self.scripts = []
        self.locations = []
        self.neighbours = neighbours

    def add_script(self, script, location):
        """Adds a script and its corresponding location to the worker's task list."""
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        """Executes all assigned scripts."""
        for (script, location) in zip(self.scripts, self.locations):
            script_data = []
            
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Gather data from the parent device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Execute the script and propagate the result to all relevant devices.
                res = script.run(script_data)
                for device in self.neighbours:
                    device.set_data(location, res)
                self.device.set_data(location, res)

    def run(self):
        """The entry point for the thread."""
        self.run_scripts()
