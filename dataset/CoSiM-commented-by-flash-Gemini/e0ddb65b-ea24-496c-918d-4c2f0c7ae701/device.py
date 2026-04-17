"""
@e0ddb65b-ea24-496c-918d-4c2f0c7ae701/device.py
@brief Distributed sensor network simulation with persistent worker pool and stateful locking.
This module implements a coordinated parallel processing framework using a 
persistent 'ThreadPool' to handle computational tasks. It utilizes a stateful 
locking protocol ('sticky locks') where data acquisition and update operations are 
intrinsically linked to maintain network-wide spatial consistency. Global temporal 
consensus is enforced through a reusable barrier.

Domain: Parallel Worker Pools, Stateful Locking, Distributed Simulation.
"""

from threading import Event, Thread, Lock
from barrier import Barrier
from thread_pool import ThreadPool


class Device(object):
    """
    Core network node representation.
    Functional Utility: Manages local data state and provides a transactional 
    interface for data access that incorporates automatic synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.barrier = None
        self.scripts = []
        self.timepoint_done = Event()

        # Spatial Mutex Pool: dedicated lock for every local sensor location.
        self.locks = {}
        for location in sensor_data:
            self.locks[location] = Lock()

        self.scripts_available = False
        # Main lifecycle management thread.
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
            barrier = Barrier(len(devices))
            self.barrier = barrier
            self.send_barrier(devices, barrier)

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
            self.scripts_available = True
        else:
            # Signal end of task assignment phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Stateful Data Retrieval.
        Functional Utility: Part 1 of a transactional update. Atomically 
        acquires the spatial lock for the location.
        @return: Sensor data or None if not found.
        """
        if location in self.sensor_data:
            # Atomic acquisition: lock is held until set_data is called.
            self.locks[location].acquire()
            return self.sensor_data[location]

        return None

    def set_data(self, location, data):
        """
        Stateful Data Update.
        Functional Utility: Part 2 of a transactional update. Replaces sensor 
        value and releases the spatial lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Release: permits other nodes or threads to access this location.
            self.locks[location].release()

    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Node-level orchestration thread.
    Functional Utility: Manages simulation timepoints and delegates tasks 
    to the persistent parallel thread pool.
    """
    
    NR_THREADS = 8

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Sized for 8 concurrent computational workers.
        self.thread_pool = ThreadPool(self.NR_THREADS)

    def run(self):
        """
        Main orchestration loop.
        Algorithm: Iterative sequence of task offloading followed by global barrier.
        """
        self.thread_pool.set_device(self.device)

        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Workload Generation.
            while True:
                # Wait for supervisor signal.
                self.device.timepoint_done.wait()
                if self.device.scripts_available:
                    self.device.scripts_available = False
                    # Offload tasks to the parallel pool.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task((neighbours, location, script))
                else:
                    # Simulation step complete: reset for next cycle.
                    self.device.timepoint_done.clear()
                    self.device.scripts_available = True
                    break

            # Synchronize: Wait for local pool to finish its tasks.
            self.thread_pool.wait()

            # Global Rendezvous point for network-wide consensus.
            self.device.barrier.wait()

        # Shutdown pool resources.
        self.thread_pool.finish()


from threading import Thread
from Queue import Queue


class ThreadPool(object):
    """
    Management layer for a persistent pool of worker threads.
    Functional Utility: Provides an asynchronous submission interface using 
    a thread-safe queue.
    """
    
    def __init__(self, nr_threads):
        self.device = None
        self.queue = Queue(nr_threads)
        self.thread_list = []
        self.create_threads(nr_threads)
        self.start_threads()

    def create_threads(self, nr_threads):
        """Spawns persistent worker threads."""
        for _ in xrange(nr_threads):
            thread = Thread(target=self.execute_task)
            self.thread_list.append(thread)

    def start_threads(self):
        """Activates the worker pool."""
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].start()

    def set_device(self, device):
        """Injects parent context."""
        self.device = device

    def submit_task(self, task):
        """Enqueues a new computational task."""
        self.queue.put(task)

    def execute_task(self):
        """
        Worker execution loop.
        Logic: Continuously pulls from the queue and applies script logic 
        until a poison pill (None) is received.
        """
        while True:
            task = self.queue.get()
            neighbours = task[0]
            script = task[2]

            # Termination Logic.
            if script is None and neighbours is None:
                self.queue.task_done()
                break

            self.run_script(task)
            self.queue.task_done()

    def run_script(self, task):
        """
        Execution logic for a single task.
        Logic: Implements the sticky lock protocol (Acquire on Read, Release on Update).
        """
        neighbours, location, script = task
        script_data = []

        # Aggregate neighborhood state.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                # Part 1: Acquire spatial lock via get_data.
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Include local sensor state.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Apply script computation.
            result = script.run(script_data)

            # Part 2: Propagation & Release via set_data.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

            self.device.set_data(location, result)

    def wait(self):
        """Blocks until the internal task queue is empty."""
        self.queue.join()

    def finish(self):
        """Gracefully shuts down the worker pool."""
        self.wait()
        for _ in xrange(len(self.thread_list)):
            self.submit_task((None, None, None))
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].join()
