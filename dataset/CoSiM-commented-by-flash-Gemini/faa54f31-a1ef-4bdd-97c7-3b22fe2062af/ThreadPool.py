"""
@faa54f31-a1ef-4bdd-97c7-3b22fe2062af/ThreadPool.py
@brief Distributed sensor network simulation with persistent worker pool and stateful locking.
This module implements a coordinated parallel processing framework using a 
persistent 'ThreadPool' to handle computational tasks asynchronously. It utilizes 
a stateful locking protocol ('sticky locks') where data acquisition and update 
operations are coupled to ensure atomic access to shared sensor locations. 
Simulation phases are coordinated through an event-driven orchestrator and a 
reusable synchronization barrier.

Domain: Parallel Worker Pools, Stateful Locking, Distributed Coordination.
"""

from Queue import Queue
from threading import Thread

class ThreadPool(object):
    """
    Management layer for a persistent pool of worker threads.
    Functional Utility: Provides an asynchronous submission interface using 
    a thread-safe queue and orchestrates the lifecycle of workers.
    """

    def __init__(self, threads_count):
        """
        Initializes the pool and spawns workers.
        @param threads_count: size of the parallel pool.
        """
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None
        self.create_workers(threads_count)
        self.start_workers()

    def create_workers(self, threads_count):
        """Spawns persistent worker threads."""
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

    def start_workers(self):
        """Activates the worker pool."""
        for thread in self.threads:
            thread.start()

    def set_device(self, device):
        """Injects parent node context."""
        self.device = device

    def execute(self):
        """
        Worker execution loop.
        Logic: Continuously pulls tasks from the queue and applies script logic 
        until a termination signal (poison pill) is received.
        """
        while True:
            # Block until a task is available.
            neighbours, script, location = self.queue.get()

            # Exit Logic: Check for poison pill (None, None).
            if neighbours is None and script is None:
                self.queue.task_done()
                return

            self.run_script(neighbours, script, location)
            # Signal task completion to support queue.join().
            self.queue.task_done()

    def run_script(self, neighbours, script, location):
        """
        Execution logic for a single task.
        Logic: Implements the sticky lock protocol (Acquire on Read, Release on Update).
        """
        script_data = []
        
        # Aggregate neighborhood state.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                # Part 1: Acquire spatial lock via get_data.
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Include local sensor data.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Apply computational logic.
            result = script.run(script_data)

            # Part 2: Propagation and release via set_data.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)
            
            self.device.set_data(location, result)

    def submit(self, neighbours, script, location):
        """Enqueues a new computational task."""
        self.queue.put((neighbours, script, location))

    def wait_threads(self):
        """Blocks until the internal task queue is completely drained."""
        self.queue.join()

    def end_threads(self):
        """
        Gracefully shuts down the worker pool.
        Logic: Flushes pending work and dispatches N poison pills.
        """
        self.wait_threads()

        # Shutdown signal for each thread.
        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)

        for thread in self.threads:
            thread.join()




from threading import Event, Thread, Lock

from barrier import Barrier
from ThreadPool import ThreadPool

class Device(object):
    """
    Core network node representation.
    Functional Utility: Manages local data state and provides a transactional 
    locking interface for spatial mutual exclusion.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.barrier = None
        # Spatial lock pool: dedicated mutex for every local sensor location.
        self.location_locks = {location : Lock() for location in sensor_data}
        self.scripts_arrived = False

        # Primary orchestration thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) initializes and distributes the shared barrier.
        """
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            self.send_barrier(devices, self.barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        """Helper to propagate the shared barrier across the network group."""
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """Injects the shared network barrier."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """Registers a computational task for the current simulation step."""
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_arrived = True
        else:
            # Signal end of task assignment phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Stateful Data Retrieval.
        Functional Utility: Part 1 of a transactional update. Atomically 
        acquires the spatial lock for the location.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Stateful Data Update.
        Functional Utility: Part 2 of a transactional update. Replaces sensor 
        value and releases the spatial lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()

    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    Node-level manager.
    Functional Utility: Orchestrates simulation phases and delegates tasks 
    to a persistent parallel worker pool.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Managed pool of 8 computational workers.
        self.thread_pool = ThreadPool(8)

    def run(self):
        """
        Main execution loop for the node orchestration.
        Algorithm: Iterative sequence of topology refresh, task offloading, and consensus.
        """
        self.thread_pool.set_device(self.device)

        while True:
            # Refresh topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Workload Generation.
            while True:
                # Wait for workload arrival or simulation step finish.
                if self.device.scripts_arrived or self.device.timepoint_done.wait():
                    if self.device.scripts_arrived:
                        self.device.scripts_arrived = False
                        # Offload tasks to the parallel pool.
                        for (script, location) in self.device.scripts:
                            self.thread_pool.submit(neighbours, script, location)
                    else:
                        # step complete: reset metadata for next cycle.
                        self.device.timepoint_done.clear()
                        self.device.scripts_arrived = True
                        break

            # Synchronize: Wait for local pool to finish its assigned workload.
            self.thread_pool.wait_threads()

            # Global Consensus rendezvous.
            self.device.barrier.wait()

        # Cleanup pool resources.
        self.thread_pool.end_threads()
