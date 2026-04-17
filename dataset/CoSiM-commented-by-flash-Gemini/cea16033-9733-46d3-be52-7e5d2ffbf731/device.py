"""
@cea16033-9733-46d3-be52-7e5d2ffbf731/device.py
@brief Distributed sensor network simulation with multi-layered parallel processing.
This module implements a sophisticated hierarchical threading model for parallel 
task execution. Each node orchestration thread (DeviceThread) delegates work to a 
transient task manager (Scripter), which in turn supervises a pool of execution workers 
(ScriptExecutor). Data consistency is enforced through fine-grained per-location 
mutexes and a two-phase semaphore-based synchronization barrier.

Domain: Hierarchical Threading, Parallel Pool Management, Fine-Grained Locking.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue


class Device(object):
    """
    Coordinator entity for a network node.
    Functional Utility: Manages local data state, fine-grained locks for sensor 
    locations, and provides the interface for hierarchical task delegation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None
        
        # Mutex to ensure atomic assignment of tasks to the queue.
        self.script_running = Lock()
        self.timepoint_done = Event()
        
        # Fine-Grained Locking: specialized mutex for every sensor location.
        self.data_locks = dict()
        self.queue = Queue()
        # concurrency boundary.
        self.available_threads = 14

        # Block Logic: Lock pool initialization.
        for loc in sensor_data:
            self.data_locks.__setitem__(loc, Lock())

        # Global access mutex for the device node.
        self.can_get_data = Lock()

        self.master = None
        self.script_over = False
        self.alive = True
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization initialization.
        Logic: Distributes the shared barrier and identifies the network coordinator.
        """
        self.barrier = ReusableBarrier(len(devices))
        self.master = devices[0]

    def assign_script(self, script, location):
        """
        Registers a task and enqueues it for parallel processing.
        Functional Utility: performs thread-safe task insertion into the worker queue.
        """
        if script is not None:
            self.script_running.acquire()
            self.scripts.append((script, location))
            self.queue.put_nowait((script, location))
            self.script_received.set()
        else:
            # Signal end of task stream for current timepoint.
            self.script_running.acquire()
            self.timepoint_done.set()

    def get_data(self, location):
        """High-level atomic retrieval of sensor data."""
        self.can_get_data.acquire()
        return_value = self.sensor_data[location] if location in self.sensor_data else None
        self.can_get_data.release()
        return return_value

    def get_device_data(self, location):
        """
        Low-level thread-safe data retrieval using spatial location locks.
        Functional Utility: ensures memory consistency during concurrent access.
        """
        if location not in self.sensor_data:
            return None

        # Lock-Coupled Access: protects specific location during read.
        self.data_locks.get(location).acquire()
        new_data = self.sensor_data[location]
        self.data_locks.get(location).release()

        return new_data

    def set_data(self, location, data):
        """
        Low-level thread-safe data update.
        Logic: Uses per-location mutex to ensure atomic modification.
        """
        if location in self.sensor_data:
            self.data_locks.get(location).acquire()
            self.sensor_data[location] = data
            self.data_locks.get(location).release()

    def shutdown(self):
        """Joins the main node manager."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Main orchestration thread for the node.
    Functional Utility: coordinates simulation phases and manages the lifecycle 
    of the Scripter manager.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation loop.
        Algorithm: Iterative task management with layered thread delegation.
        """
        while True:
            # Secure node for topology update.
            self.device.can_get_data.acquire()
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # Synchronization Point: ensure all devices exit clean.
                self.device.master.barrier.wait()
                self.device.can_get_data.release()
                return

            # Delegate task management to a specialized manager thread.
            script_instance = Scripter(self.device, neighbours)
            script_instance.start()

            # Wait for supervisor signal indicating all tasks are dispatched.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Lifecycle Management: Signal termination to the worker layer.
            self.device.script_over = True
            self.device.script_received.set()

            # Reclaim manager thread.
            script_instance.join()

            # Task Buffer Reset.
            for (script, location) in self.device.scripts:
                self.device.queue.put_nowait((script, location))

            self.device.script_over = False

            # Global Temporal Consensus.
            self.device.master.barrier.wait()

            self.device.can_get_data.release()
            self.device.script_running.release()


class Scripter(Thread):
    """
    Layer 2 Manager: Pool Controller.
    Functional Utility: Manages a pool of execution workers and coordinates 
    the dispatch of computational scripts.
    """

    def __init__(self, device, neighbours):
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        Manager execution loop.
        Logic: Spawns 13 executors and waits for termination signal from the 
        DeviceThread layer.
        """
        list_executors = []

        # Spawns Layer 3: Execution Workers.
        for iterator in range(1, self.device.available_threads):
            executor = ScriptExecutor(self.device, self.device.queue, self.neighbours, iterator)
            list_executors.append(executor)
            executor.start()

        while True:
            # Wait for dispatch triggers.
            self.device.script_received.wait()
            self.device.script_received.clear()

            if self.device.script_over:
                # Poison Pill Dispatch: Shutdown worker pool.
                for iterator in range(1, self.device.available_threads):
                    self.device.queue.put((None, None))

                # Reclaim worker resources.
                for executor in list_executors:
                    executor.join()

                # Refresh queue for next simulation timepoint.
                self.device.queue = Queue(-1)
                return

            self.device.script_running.release()


class ScriptExecutor(Thread):
    """
    Layer 3: Execution Worker.
    Functional Utility: Consumes individual script tasks and implements the 
    core processing logic with neighborhood data exchange.
    """

    def __init__(self, device, queue, neighbours, identifier):
        Thread.__init__(self, name="Script Executor %d" % identifier)
        self.device = device
        self.queue = queue
        self.neighbours = neighbours

    def run(self):
        """
        Worker execution loop.
        Logic: Continuous 'pull-process' cycle from the shared device queue.
        """
        while True:
            # Block on task arrival.
            (script, location) = self.queue.get()
            # Check for termination signal.
            if script is None:
                return

            script_data = []
            # Aggregate neighborhood state using fine-grained locks.
            for device in self.neighbours:
                data = device.get_device_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Include local sensor state.
            data = self.device.get_device_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Apply computational logic.
                result = script.run(script_data)
                
                # Propagation: Atomic updates to the neighborhood graph.
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)


class ReusableBarrier:
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Orchestrates a global rendezvous point for all devices.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Double-gate synchronization logic."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counting and semaphore release."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Threshold reached: release all participants.
                for iterator in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
