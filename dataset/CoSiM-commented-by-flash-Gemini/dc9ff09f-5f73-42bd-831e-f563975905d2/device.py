"""
@dc9ff09f-5f73-42bd-831e-f563975905d2/device.py
@brief Distributed sensor network simulation with hardware-aware parallel worker pool.
This module implements a dynamic processing model that optimizes parallelism by 
sizing the internal worker pool based on the system's CPU core count. Each node 
designated a coordinator thread (ID 0) to manage topology discovery and task 
distribution, while the remaining pool consumes tasks from a thread-safe queue. 
Consistency is enforced via a global spatial lock pool (implemented as semaphores) 
and a robust multi-stage barrier synchronization.

Domain: Hardware-Optimized Parallelism, Worker Pools, Semaphore-Based Spatial Locking.
"""

from threading import Event, Thread, Lock, Semaphore
from multiprocessing import cpu_count
from Queue import Queue


class ReusableBarrierSem(object):
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Uses a double-gate mechanism with semaphores to ensure 
    perfect temporal alignment across a fixed set of threads.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: Number of participants in the rendezvous.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Orchestrates the two-phase synchronization cycle."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """Arrival phase: block until all participants have arrived."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Exit phase: prevent overtaking before group clearance."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Coordinates simulation step orchestration and manages 
    a persistent pool of parallel workers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.neighbours = []

        # Optimization Logic: sizes the worker pool based on hardware capability.
        self.num_of_threads = cpu_count()
        if self.num_of_threads < 8:
            self.num_of_threads = 8

        # Core Task Queue and signaling mechanism.
        self.tasks = Queue()
        self.semaphore = Semaphore(0)

        # Spatial Locking: Mutex pool for every sensor location in the network.
        self.num_locations = self.supervisor.supervisor.testcase.num_locations
        self.lock_locations = []

        # Shared state update protection.
        self.lock_queue = Lock()
        # Internal barrier for local thread group.
        self.barrier = ReusableBarrierSem(self.num_of_threads)
        # External barrier for network-wide consensus.
        self.global_barrier = ReusableBarrierSem(0)
        # Functional Utility: worker pool manager.
        self.pool = Pool(self)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) initializes the network-wide barrier 
        and the global pool of spatial semaphores.
        """
        if self.device_id == 0:
            self.global_barrier = ReusableBarrierSem(len(devices))
            # Atomic Resource Allocation: creates a mutex for each sensor location.
            for _ in range(self.num_locations):
                self.lock_locations.append(Semaphore(1))
            # Propagation: Distribute resources to peer nodes.
            for device in devices:
                device.global_barrier = self.global_barrier
                device.lock_locations = self.lock_locations

    def assign_script(self, script, location):
        """Registers a computational task for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal end of task assignment phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] \
            if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully terminates the worker pool."""
        self.pool.shutdown()


class Pool(object):
    """
    Management layer for persistent worker threads.
    Functional Utility: Provides an asynchronous submission interface and 
    handles thread lifecycle.
    """

    def __init__(self, device):
        self.device = device
        self.thread_list = []
        # Spawns a pool of hardware-optimized worker threads.
        for i in range(self.device.num_of_threads):
            self.thread_list.append(DeviceThread(self.device, i))
        for thread in self.thread_list:
            thread.start()

    def add_task(self, task):
        """Enqueues a new script for execution and signals idle workers."""
        self.device.tasks.put(task)
        self.device.semaphore.release()

    def shutdown(self):
        """Joins all threads in the pool."""
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    """
    Simulated worker implementation with role-based coordination.
    Functional Utility: Executes computational scripts from the shared node queue 
    while participating in both local and global simulation phases.
    """

    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id
        self.script = None
        self.location = None
        self.script_data = []
        self.data = None
        self.result = None

    def run(self):
        """
        Main worker execution loop.
        Algorithm: Iterative multi-phase synchronization: 
        Coordination -> Local Barrier -> Task Consumption -> Global Barrier.
        """
        while True:
            # Coordination Phase: Only thread 0 handles topology discovery.
            if self.thread_id == 0:
                self.device.timepoint_done.clear()
                self.device.neighbours = \
                                self.device.supervisor.get_neighbours()

            # Local Synchronization: ensure all threads are aligned for the next step.
            self.device.barrier.wait()

            # Exit Logic.
            if self.device.neighbours is None:
                break

            # Block Logic: Workload Generation (Thread 0 only).
            if self.thread_id == 0:
                # Wait for supervisor to finish assignments.
                self.device.timepoint_done.wait()
                # Offload all scripts into the local worker queue.
                for (script, location) in self.device.scripts:
                    self.device.pool.add_task((script, location))
                # Release all workers from the arrival semaphore.
                for _ in range(self.device.num_of_threads):
                    self.device.semaphore.release()

            # Block Logic: Task Consumption Phase.
            # Threads pull and process tasks until the local queue is drained.
            while True:
                # Wait for available task signal.
                self.device.semaphore.acquire()
                with self.device.lock_queue:
                    if not self.device.tasks.empty():
                        (self.script, self.location) = self.device.tasks.get()
                    else:
                        break

                # Spatial Synchronization: acquire exclusive access to the sensor location.
                self.device.lock_locations[self.location].acquire()
                self.script_data = []
                # Aggregate neighborhood data.
                for device in self.device.neighbours:
                    self.data = device.get_data(self.location)
                    if self.data is not None:
                        self.script_data.append(self.data)
                
                # Include local node data.
                self.data = self.device.get_data(self.location)
                if self.data is not None:
                    self.script_data.append(self.data)

                if self.script_data != []:
                    # Compute result and propagate to neighbors and self.
                    self.result = self.script.run(self.script_data)
                    for device in self.device.neighbours:
                        device.set_data(self.location, self.result)
                    self.device.set_data(self.location, self.result)

                # Release the spatial mutex.
                self.device.lock_locations[self.location].release()

            # Global Phase: Coordinator thread waits for network-wide temporal alignment.
            if self.thread_id == 0:
                self.device.global_barrier.wait()

            # Final Local Synchronization before next simulation step.
            self.device.barrier.wait()
