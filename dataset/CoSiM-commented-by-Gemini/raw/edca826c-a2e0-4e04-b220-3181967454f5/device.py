"""
This module implements a device simulation using a worker-pool architecture with a
load-balancing strategy for script distribution.

Each `Device` has a main `DeviceThread` that, for each time step, creates a fixed
pool of `DeviceWorker` threads. Scripts are then assigned to these workers based
on location affinity and load, a more sophisticated approach to task distribution
than seen in other versions. The implementation appears to be for Python 2, given
the use of `Queue.Queue`.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue # Note: In Python 3, this would be `import queue`

class ReusableBarrier(object):
    """A custom, two-phase reusable barrier implemented with Semaphores."""
    def __init__(self, num_threads):
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
    """
    Represents a device node that distributes work to a pool of worker threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.result_queue = Queue.Queue()
        # WARNING: A lock is used for setting data, but not for getting it,
        # creating a potential race condition between readers and writers.
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
        """Initializes and distributes shared resources from the coordinator device."""
        if self.device_id == devices[0].device_id:
            self.neighbours_lock = Lock()
            self.neighbours_barrier = ReusableBarrier(len(devices))
        else:
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to the device for the next time step."""
        if script is not None:
            self.scripts.append((script, location))
        # The script_received event is set regardless, signaling the main thread
        # to check the script list. A None script also signals timepoint completion.
        self.script_received.set()
        if script is None:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data for a given location.
        WARNING: This method is not thread-safe.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets data for a given location in a thread-safe manner.
        """
        self.set_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_lock.release()

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    The main orchestration thread for a device. It manages a pool of worker threads
    and implements a load-balancing strategy for script assignment.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []

    def run(self):
        """Main simulation loop."""
        while True:
            # Acquire a global lock to get neighbor info, suggesting the
            # supervisor might not be thread-safe.
            self.device.neighbours_lock.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.neighbours_lock.release()

            if neighbours is None:
                break # Terminate signal.

            self.device.script_received.wait()

            # --- Worker Pool and Load Balancing ---
            # 1. Create a fixed pool of 8 worker threads for this time step.
            self.workers = [DeviceWorker(self.device, i, neighbours) for i in range(8)]

            # 2. Assign scripts to workers using a load-balancing strategy.
            for (script, location) in self.device.scripts:
                added = False
                # First, try to find a worker already handling this location (location affinity).
                for worker in self.workers:
                    if location in worker.locations:
                        worker.add_script(script, location)
                        added = True
                        break # Found a worker, move to the next script.

                # If no worker has this location, find the worker with the fewest locations.
                if not added:
                    chosen_worker = min(self.workers, key=lambda w: len(w.locations))
                    chosen_worker.add_script(script, location)

            # 3. Start all workers to run their assigned scripts in parallel.
            for worker in self.workers:
                worker.start()

            # 4. Wait for all workers to complete.
            for worker in self.workers:
                worker.join()

            # 5. Synchronize with all other devices before the next time step.
            self.device.neighbours_barrier.wait()
            self.device.script_received.clear()


class DeviceWorker(Thread):
    """A worker thread that executes a batch of scripts sequentially."""
    def __init__(self, device, worker_id, neighbours):
        Thread.__init__(self)
        self.device = device
        self.worker_id = worker_id
        self.scripts = []
        self.locations = []
        self.neighbours = neighbours

    def add_script(self, script, location):
        """Adds a script and its location to this worker's queue."""
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        """
        Executes all scripts assigned to this worker.
        Concurrency control is implicit: since one worker runs this loop, scripts
        on this worker that share a location will not run concurrently.
        """
        for (script, location) in zip(self.scripts, self.locations):
            script_data = []
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Gather data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                res = script.run(script_data)
                # Propagate results to neighbors and the local device.
                for device in self.neighbours:
                    device.set_data(location, res)
                self.device.set_data(location, res)

    def run(self):
        """The thread's entry point."""
        self.run_scripts()