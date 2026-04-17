"""
@d277a1f8-ae8e-46e0-8b4e-0e633cb6b221/device.py
@brief Distributed sensor network simulation with a centralized global worker pool.
This module implements a unique architecture where individual network nodes act 
as task producers, offloading computational scripts to a shared, global 'WorkPool'. 
This centralized processing engine handles parallel execution for all devices in 
the network simultaneously. Consistency is enforced through global spatial locks 
and cross-node synchronization barriers.

Domain: Centralized Parallelism, Shared Worker Pools, Distributed State Management.
"""

from threading import Event, Thread, Lock
from pool import WorkPool
from reusable_barrier import ReusableBarrier

class Device(object):
    """
    Simulated network node acting as a task producer.
    Functional Utility: Manages local data state and submits computational tasks 
    to a network-wide shared worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()
        self.scripts = []

        self.timepoint_done = Event()
        self.other_devs = []
        # Mutex for protecting local data access.
        self.slock = Lock()

        self.barrier = None
        self.process = Event()

        # Shared Resources: populated during setup_devices.
        self.global_thread_pool = None
        self.glocks = {}

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global resource initialization and distribution.
        Logic: Node 0 initializes the shared WorkPool, Barrier, and global lock pool. 
        Subsequent nodes register their locations and attach to the shared resources.
        """
        self.other_devs = devices
        # Leader Node Logic.
        if self.device_id == self.other_devs[0].device_id:
            locks = {}
            for loc in self.sensor_data:
                locks[loc] = Lock()
            dev_cnt = len(devices)
            self.glocks = locks
            self.barrier = ReusableBarrier(dev_cnt)
            # Central Processing Engine: sized for 16 concurrent workers.
            self.global_thread_pool = WorkPool(16)
        else:
            # Participant Node Logic: registers local locations in the global map.
            for loc in self.sensor_data:
                self.other_devs[0].glocks[loc] = Lock()
            
            # Attach to the shared network resources.
            self.glocks = self.other_devs[0].glocks
            self.global_thread_pool = self.other_devs[0].global_thread_pool
            self.barrier = self.other_devs[0].barrier

    def assign_script(self, script, location):
        """Registers a task and signals the orchestration thread."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        ret = None
        with self.slock:
            if location in self.sensor_data:
                ret = self.sensor_data[location]
        return ret

    def set_data(self, location, data):
        """Thread-safe update of local state."""
        with self.slock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Joins the node management thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    Node-level lifecycle coordinator.
    Functional Utility: Manages the submission of local tasks to the global pool 
    and handles simulation phase transitions.
    """
    
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main orchestration loop.
        Algorithm: Iterative task offloading with barrier-based synchronization.
        """
        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Termination Logic: leader node triggers pool shutdown.
                if self.device.device_id is self.device.other_devs[0].device_id:
                    self.device.global_thread_pool.end()
                break

            # Wait for supervisor to finish script assignments.
            self.device.script_received.wait()

            # Offload Logic: push all local scripts to the global WorkPool.
            for (script, location) in self.device.scripts:
                self.device.global_thread_pool.work((self.device,
                									script,
                									location,
                									neighbours))

            # Block until the global pool finishes this node's tasks.
            self.device.global_thread_pool.finish_work()
            self.device.script_received.clear()
            # Synchronize with the entire network.
            self.device.barrier.wait()


from threading import Lock, Event, Semaphore, Thread

class WorkerThread(Thread):
    """
    Global worker implementation.
    Functional Utility: Consumes tasks from a multi-node shared queue and 
    executes them while maintaining spatial mutual exclusion across the network.
    """
    
    def __init__(self, i, parent_work_pool):
        Thread.__init__(self, name="WorkerThread%d" % i)
        self.pool = parent_work_pool

    def run(self):
        """
        Main worker execution loop.
        Algorithm: Pull-process cycle with network-wide spatial locking.
        """
        while True:
            # Wait for task arrival or shutdown signal.
            self.pool.task_sign.acquire()
            if self.pool.stop:
                break
            
            current_task = (None, None, None, None)
            # Atomic task retrieval from the shared list.
            with self.pool.task_lock:
                task_count = len(self.pool.tasks_list)
                if task_count > 0:
                    current_task = self.pool.tasks_list[0]
                    self.pool.tasks_list = self.pool.tasks_list[1:]
                
                # Signal if the queue has become empty.
                if task_count == 1:
                    self.pool.no_tasks.set()

            if current_task is not None:
                (current_device, script, location, neighbourhood) = current_task
                
                # Global Critical Section: ensures network-wide consistency for the location.
                with current_device.glocks[location]:
                    common_data = []
                    
                    # Neighborhood aggregation.
                    for neighbour in neighbourhood:
                        data = neighbour.get_data(location)
                        if data is not None:
                            common_data.append(data)
                    
                    # Owner node state integration.
                    data = current_device.get_data(location)
                    if data is not None:
                        common_data.append(data)

                    if common_data != []:
                        # Execute computation and propagate results.
                        result = script.run(common_data)
                        for neighbour in neighbourhood:
                            neighbour.set_data(location, result)
                        current_device.set_data(location, result)

class WorkPool(object):
    """
    Centralized task management hub.
    Functional Utility: Implements an asynchronous task queue shared by 
    multiple network nodes, managing worker lifecycles and signaling.
    """
    
    def __init__(self, size):
        self.size = size
        self.tasks_list = [] 
        self.task_lock = Lock()
        # semaphore acts as a counter for available tasks.
        self.task_sign = Semaphore(0)
        # Event used to signal queue drainage.
        self.no_tasks = Event()
        self.no_tasks.set()
        self.stop = False

        # Spawn persistent worker pool.
        self.workers = []
        for i in xrange(self.size):
            worker = WorkerThread(i, self)
            self.workers.append(worker)

        for worker in self.workers:
            worker.start()

    def work(self, task):
        """Submits a new task to the global queue."""
        with self.task_lock:
            self.tasks_list.append(task)
            self.task_sign.release()
            if self.no_tasks.is_set():
                self.no_tasks.clear()

    def finish_work(self):
        """Blocks until the global queue is drained."""
        self.no_tasks.wait()

    def end(self):
        """Gracefully terminates all global worker threads."""
        self.finish_work()
        self.stop = True
        # release all workers to allow them to check the stop flag.
        for thread in self.workers:
            self.task_sign.release()
        for thread in self.workers:
            thread.join()
