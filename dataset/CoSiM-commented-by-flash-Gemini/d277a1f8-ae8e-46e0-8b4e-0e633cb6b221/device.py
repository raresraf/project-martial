"""
@d277a1f8-ae8e-46e0-8b4e-0e633cb6b221/device.py
@brief Distributed sensor processing simulation using a global thread pool and centralized synchronization.
* Algorithm: Collective task offloading to a system-wide pool of 16 worker threads with per-location locking and phased barriers.
* Functional Utility: Orchestrates simulation timepoints across multiple devices by managing a shared worker resource and ensuring consistent distributed state updates.
"""

from threading import Event, Thread, Lock
from pool import WorkPool
from reusable_barrier import ReusableBarrier

class Device(object):
    """
    @brief Encapsulates a sensor node that manages its local readings and interacts with the global pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and bootstraps the coordination thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()
        self.scripts = []

        self.timepoint_done = Event()
        self.other_devs = []
        self.slock = Lock() # Intent: Serializes access to local sensor data.

        self.barrier = None
        self.process = Event()

        self.global_thread_pool = None
        self.glocks = {} # Intent: Maps global locations to synchronization locks.

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization and pool initialization.
        Invariant: Root device (index 0) initializes the shared WorkPool and collective locks.
        """
        self.other_devs = devices
        if self.device_id == self.other_devs[0].device_id:
            locks = {}
            for loc in self.sensor_data:
                locks[loc] = Lock()
            dev_cnt = len(devices)
            self.glocks = locks
            self.barrier = ReusableBarrier(dev_cnt)
            # Domain: Cluster-wide Resource Scaling - 16 threads for all participating devices.
            self.global_thread_pool = WorkPool(16)
        else:
            # Logic: Peers register their local locations in the root's global lock map.
            for loc in self.sensor_data:
                self.other_devs[0].glocks[loc] = Lock()
            self.glocks = self.other_devs[0].glocks
            self.global_thread_pool = self.other_devs[0].global_thread_pool
            self.barrier = self.other_devs[0].barrier

    def assign_script(self, script, location):
        """
        @brief Buffers an incoming processing task.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signals that all scripts for the current phase have arrived.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Synchronized retrieval of sensor data.
        """
        ret = None
        with self.slock:
            if location in self.sensor_data:
                ret = self.sensor_data[location]
        return ret

    def set_data(self, location, data):
        """
        @brief Synchronized update of sensor data.
        """
        with self.slock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Terminates the device management thread.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief Coordination thread managing the lifecycle of a single device node.
    Algorithm: Offloads local script batch to the global pool and waits for collective alignment.
    """
    
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main coordination loop for the device.
        """
        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Logic: Final shutdown - only the root device triggers pool termination.
                if self.device.device_id is self.device.other_devs[0].device_id:
                    self.device.global_thread_pool.end()
                break

            # Block Logic: Waits for script delivery start.
            self.device.script_received.wait()

            # Dispatch Phase: Submits all assigned scripts to the shared global WorkPool.
            for (script, location) in self.device.scripts:
                self.device.global_thread_pool.work((self.device,
                									script,
                									location,
                									neighbours))

            # Synchronization Phase 1: Wait for local tasks in the pool to complete.
            self.device.global_thread_pool.finish_work()
            self.device.script_received.clear()
            
            # Synchronization Phase 2: Wait for cluster-wide alignment.
            self.device.barrier.wait()


from threading import Lock, Event, Semaphore, Thread

class WorkerThread(Thread):
    """
    @brief Persistent worker thread within the global WorkPool.
    """
    
    def __init__(self, i, parent_work_pool):
        Thread.__init__(self, name="WorkerThread%d" % i)
        self.pool = parent_work_pool

    def run(self):
        """
        @brief Main loop for processing individual script tasks from the global queue.
        Algorithm: Producer-Consumer consumption with location-based mutual exclusion.
        """
        while True:
            # Logic: Blocks until a task signal is received.
            self.pool.task_sign.acquire()
            if self.pool.stop:
                break
            
            current_task = (None, None, None, None)
            # Invariant: Atomic removal of the head task from the shared list.
            with self.pool.task_lock:
                task_count = len(self.pool.tasks_list)
                if task_count > 0:
                    current_task = self.pool.tasks_list[0]
                    self.pool.tasks_list = self.pool.tasks_list[1:]
                
                if task_count == 1:
                    # Logic: Signals that the pool is now empty.
                    self.pool.no_tasks.set()

            if current_task is not None:
                (current_device, script, location, neighbourhood) = current_task
                # Pre-condition: Must acquire global location lock for atomic distributed update.
                with current_device.glocks[location]:
                    common_data = []
                    
                    # Distributed Aggregation: Collect readings from neighbors and self.
                    for neighbour in neighbourhood:
                        data = neighbour.get_data(location)
                        if data is not None:
                            common_data.append(data)
                    
                    data = current_device.get_data(location)
                    if data is not None:
                        common_data.append(data)

                    # Execution and Propagation Phase.
                    if common_data != []:
                        result = script.run(common_data)
                        for neighbour in neighbourhood:
                            neighbour.set_data(location, result)
                        
                        current_device.set_data(location, result)

class WorkPool(object):
    """
    @brief Shared thread pool manager for simulation-wide script execution.
    """
    
    def __init__(self, size):
        """
        @brief Bootstraps the persistent worker pool.
        """
        self.size = size
        self.tasks_list = [] 
        self.task_lock = Lock()
        self.task_sign = Semaphore(0) # Intent: Counting semaphore for available tasks.
        self.no_tasks = Event()
        self.no_tasks.set()
        self.stop = False

        self.workers = []
        for i in xrange(self.size):
            worker = WorkerThread(i, self)
            self.workers.append(worker)

        for worker in self.workers:
            worker.start()

    def work(self, task):
        """
        @brief Enqueues a new simulation task into the global pool.
        """
        with self.task_lock:
            self.tasks_list.append(task)
            self.task_sign.release()
            if self.no_tasks.is_set():
                self.no_tasks.clear()

    def finish_work(self):
        """
        @brief Blocks until all enqueued tasks have been consumed by workers.
        """
        self.no_tasks.wait()

    def end(self):
        """
        @brief Orchestrates a graceful shutdown of all workers in the pool.
        """
        self.finish_work()
        self.stop = True
        # Logic: Wakes all workers to trigger shutdown check.
        for thread in self.workers:
            self.task_sign.release()
        for thread in self.workers:
            thread.join()
