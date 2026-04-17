"""
@ca18b67f-6cdd-466f-a8eb-a2875829f843/device.py
@brief Distributed sensor network simulation with affinity-based load balancing.
This module implements a sophisticated task distribution model where computational 
scripts are assigned to a pool of persistent workers using a spatial affinity heuristic. 
By routing tasks targeting the same sensor location to the same worker thread, 
the system minimizes local lock contention and optimizes data access patterns. 
Global temporal consistency is enforced via a two-phase semaphore barrier.

Domain: Load Balancing, Spatial Affinity, Parallel Worker Pools.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue

class ReusableBarrier(object):
    """
    Two-phase reusable barrier implementation using semaphores.
    Functional Utility: Provides a robust synchronization point for a fixed group 
    of threads, ensuring clean phase transitions through a double-gate mechanism.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: Number of participating threads.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Orchestrates the two-phase arrival and exit sequence."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Internal phase logic using atomic counter decrement and semaphore signaling.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Arrival threshold met: release all threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for immediate reuse.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local sensor data and coordinates global 
    synchronization primitives across the device group.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.result_queue = Queue.Queue()
        # Mutex for protecting local data state updates.
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
        """
        Global resource distribution.
        Logic: Node 0 acts as a singleton factory for shared locks and barriers 
        which are then distributed to all peer devices.
        """
        if self.device_id == devices[0].device_id:
            self.neighbours_lock = Lock()
            self.neighbours_barrier = ReusableBarrier(len(devices))
        else:
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier

        self.thread.start()

    def assign_script(self, script, location):
        """Registers a task and signals completion of the simulation step assignment."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Thread-safe update of local sensor state.
        Functional Utility: uses a dedicated mutex to ensure atomic value replacement.
        """
        self.set_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_lock.release()

    def shutdown(self):
        """Joins the node's management thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Management thread for the node simulation.
    Functional Utility: Implements a load-balancing task distributor that 
    respects spatial location affinity.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []

    def run(self):
        """
        Main simulation loop.
        Algorithm: Iterative task partitioning and worker pool management.
        """
        while True:
            # Topology Discovery Phase.
            self.device.neighbours_lock.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.neighbours_lock.release()

            if neighbours is None:
                break

            # Synchronize on task availability.
            self.device.script_received.wait()

            # Initialize a pool of 8 transient workers for the current timepoint.
            self.workers = []
            for i in range(8):
                self.workers.append(DeviceWorker(self.device, i, neighbours))

            # Block Logic: Load Balancing with Spatial Affinity.
            # Logic: Attempts to route tasks for the same location to the same worker.
            for (script, location) in self.device.scripts:
                added = False
                for worker in self.workers:
                    if location in worker.locations:
                        worker.add_script(script, location)
                        added = True

                # Heuristic: If no affinity found, pick the least-loaded worker.
                if added == False:
                    minimum = len(self.workers[0].locations)
                    chosen_worker = self.workers[0]
                    for worker in self.workers:
                        if minimum > len(worker.locations):
                            minimum = len(worker.locations)
                            chosen_worker = worker

                    chosen_worker.add_script(script, location)

            # Parallel Execution Phase.
            for worker in self.workers:
                worker.start()

            # Barrier Point: Wait for local pool completion.
            for worker in self.workers:
                worker.join()

            # Global Barrier Point: Ensure network-wide temporal consistency.
            self.device.neighbours_barrier.wait()
            self.device.script_received.clear()


class DeviceWorker(Thread):
    """
    Transient worker thread for batch script execution.
    Functional Utility: Processes a subset of scripts assigned by the DeviceThread.
    """

    def __init__(self, device, worker_id, neighbours):
        Thread.__init__(self)
        self.device = device
        self.worker_id = worker_id
        self.scripts = []
        self.locations = []
        self.neighbours = neighbours

    def add_script(self, script, location):
        """Queues a task for batch processing."""
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        """
        Execution loop for assigned tasks.
        Logic: Performs neighborhood aggregation and updates neighbors in parallel.
        """
        for (script, location) in zip(self.scripts, self.locations):
            script_data = []
            
            # Aggregate neighborhood state.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Include own state.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Apply computational logic.
                res = script.run(script_data)

                # Propagation: Update all peers in the neighborhood graph.
                for device in self.neighbours:
                    device.set_data(location, res)
                self.device.set_data(location, res)

    def run(self):
        self.run_scripts()
