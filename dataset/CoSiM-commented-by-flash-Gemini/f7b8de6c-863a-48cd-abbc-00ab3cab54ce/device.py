"""
@f7b8de6c-863a-48cd-abbc-00ab3cab54ce/device.py
@brief Distributed sensor network simulation with persistent worker pool and spatial locking.
This module implements a coordinated processing framework where a managed pool of 
8 persistent worker threads (Worker) executes computational tasks via a shared 
queue. Consistency is guaranteed through a network-wide mapping of spatial locks 
for individual sensor locations, while global temporal alignment is enforced 
via a monitor-based reusable barrier.

Domain: Parallel Worker Pools, Spatial Mutual Exclusion, Distributed Coordination.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from thread_pool import ThreadPool

class Device(object):
    """
    Representation of a node in the sensor network simulation.
    Functional Utility: Manages local data state, coordinates the distribution 
    of global synchronization resources, and maintains a local parallel worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        # Shared synchronization resources populated during setup.
        self.barrier = None
        # Spatial lock repository: map of location indices to mutexes.
        self.location_locks = {}

        # Primary node management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (first to run setup) initializes the shared barrier 
        and discovers all spatial locations across the network to pre-allocate locks.
        """
        if self.barrier is None:
            # Singleton setup: ensure all nodes share the same rendezvous point.
            self.barrier = ReusableBarrierCond(len(devices))

            # Block Logic: Spatial Lock discovery and propagation.
            for device in devices:
                device.barrier = self.barrier
                
                # Discovery: pre-allocate a mutex for every sensor location managed by this group.
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
                
                # Distribution: propagate the global lock map.
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """Registers a task and signals completion of the assignment phase."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Finalize workload for current step.
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Node-level task orchestrator.
    Functional Utility: Manages simulation timepoints and delegates computational 
    scripts to a persistent parallel thread pool.
    """
    
    # Degree of local parallelism.
    NO_CORES = 8

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # managed pool of computational workers.
        self.thread_pool = ThreadPool(self.device, DeviceThread.NO_CORES)

    def run(self):
        """
        Main orchestration loop.
        Algorithm: Iterative sequence: Wait for step -> Dispatch tasks -> Consensus.
        """
        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Termination Logic: dispatch poison pills to all workers.
                for _ in xrange(DeviceThread.NO_CORES):
                    self.thread_pool.submit_task(None, None, None)
                
                self.thread_pool.end_workers()
                break

            # Block until assignment phase for the step is complete.
            self.device.timepoint_done.wait()

            # Dispatch workload to the parallel workers.
            for (script, location) in self.device.scripts:
                self.thread_pool.submit_task(script, location, neighbours)

            # Phase Reset.
            self.device.timepoint_done.clear()

            # Global Consensus rendezvous.
            self.device.barrier.wait()




from threading import Thread
from Queue import Queue

class Worker(Thread):
    """
    Persistent computational worker.
    Functional Utility: Consumes individual script tasks and implements the 
    core processing logic with spatial mutual exclusion.
    """

    def __init__(self, device, task_queue):
        Thread.__init__(self)
        self.device = device
        self.task_queue = task_queue

    def run(self):
        """
        Worker execution loop.
        Logic: Continuous 'pull-process' cycle from the shared node queue.
        """
        while True:
            # Block on task arrival.
            script, location, neighbours = self.task_queue.get()

            # Termination check.
            if (script is None and location is None and neighbours is None):
                self.task_queue.task_done()
                break

            # Critical Section: Spatial mutual exclusion across the entire network.
            with self.device.location_locks[location]:
                self.run_task(script, location, neighbours)

            # Signal task completion to support queue.join().
            self.task_queue.task_done()

    def run_task(self, script, location, neighbours):
        """
        Computational Logic.
        Algorithm: Aggregate neighborhood data and propagate results.
        """
        script_data = []
        # Aggregate neighborhood state.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Include local sensor data.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Apply computational logic and propagate results to all nodes in the graph.
            result = script.run(script_data)

            for device in neighbours:
                device.set_data(location, result)

            self.device.set_data(location, result)


class ThreadPool(object):
    """
    Management layer for a persistent pool of worker threads.
    Functional Utility: Provides an asynchronous submission interface using 
    a thread-safe queue.
    """

    def __init__(self, device, no_workers):
        self.device = device
        self.no_workers = no_workers
        
        # Task distribution buffer.
        self.task_queue = Queue(no_workers)
        self.workers = []
        self.initialize_workers()

    def initialize_workers(self):
        """Spawns and activates persistent worker threads."""
        for _ in xrange(self.no_workers):
            self.workers.append(Worker(self.device, self.task_queue))

        for worker in self.workers:
            worker.start()

    def end_workers(self):
        """Gracefully terminates all pool threads."""
        self.task_queue.join()

        for worker in self.workers:
            worker.join()

    def submit_task(self, script, location, neighbours):
        """Enqueues a new task for parallel execution."""
        self.task_queue.put((script, location, neighbours))
