"""
This module implements a device simulation using a single, globally shared thread
pool to process all computational tasks from all devices.

The architecture consists of a custom `WorkPool` of `WorkerThread`s that is
created by a master device and shared among all other devices. Each device's
main `DeviceThread` acts as a dispatcher, adding tasks to this global pool.
While the worker logic itself is sound, the overall architecture contains
significant flaws that lead to serialized execution instead of parallel processing
across devices.
"""
from threading import Event, Thread, Lock, Semaphore
from pool import WorkPool
from reusable_barrier import ReusableBarrier


class Device(object):
    """
    Represents a device that submits tasks to a global, shared thread pool.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.other_devs = []
        # A lock to protect this device's own sensor_data dictionary.
        self.slock = Lock()
        # Shared resources to be initialized by the master device.
        self.barrier = None
        self.process = Event()
        self.global_thread_pool = None
        self.glocks = {} # Shared dictionary of location-based locks.

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (barrier, locks, thread pool).

        NOTE: This setup contains a race condition. Slave devices add locks to the
        shared `glocks` dictionary without any synchronization, which is not safe.
        """
        self.other_devs = devices
        # The first device in the list acts as the master.
        if self.device_id == self.other_devs[0].device_id:
            locks = {}
            for loc in self.sensor_data:
                locks[loc] = Lock()
            dev_cnt = len(devices)
            self.glocks = locks
            self.barrier = ReusableBarrier(dev_cnt)
            # The master device creates the single, globally shared thread pool.
            self.global_thread_pool = WorkPool(16)
        else: # Slave devices
            # RACE CONDITION: Multiple slave devices might write to this dict concurrently.
            for loc in self.sensor_data:
                self.other_devs[0].glocks[loc] = Lock()
            # Slaves get a reference to the master's shared objects.
            self.glocks = self.other_devs[0].glocks
            self.global_thread_pool = self.other_devs[0].global_thread_pool
            self.barrier = self.other_devs[0].barrier

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Thread-safely gets data from the local sensor dictionary."""
        ret = None
        with self.slock:
            if location in self.sensor_data:
                ret = self.sensor_data[location]
        return ret

    def set_data(self, location, data):
        """Thread-safely sets data in the local sensor dictionary."""
        with self.slock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class DeviceThread(Thread):
    """
    The main dispatcher thread for a device. It adds tasks to the global pool.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop that dispatches tasks and waits for completion.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Only the master device should shut down the shared pool.
                if self.device.device_id is self.device.other_devs[0].device_id:
                    self.device.global_thread_pool.end()
                break

            self.device.script_received.wait()

            # Dispatch all assigned scripts as tasks to the global thread pool.
            for (script, location) in self.device.scripts:
                self.device.global_thread_pool.work((self.device,
                									script,
                									location,
                									neighbours))

            # --- CRITICAL FLAW ---
            # Calling finish_work() on the SHARED pool from EACH device thread
            # serializes the simulation. This call blocks until ALL tasks from
            # ALL devices are done, preventing true parallel execution between devices.
            self.device.global_thread_pool.finish_work()
            self.device.script_received.clear()
            self.device.barrier.wait()


class WorkerThread(Thread):
    """
    A persistent worker thread managed by the WorkPool.
    """
    def __init__(self, i, parent_work_pool):
        Thread.__init__(self, name="WorkerThread%d" % i)
        self.pool = parent_work_pool

    def run(self):
        """
        Continuously fetches tasks from the pool's queue and executes them.
        """
        while True:
            # Blocks until a task is available.
            self.pool.task_sign.acquire()
            if self.pool.stop:
                break # Exit loop on shutdown signal.

            current_task = (None, None, None, None)
            # Safely pop a task from the shared list.
            with self.pool.task_lock:
                task_count = len(self.pool.tasks_list)
                if task_count > 0:
                    current_task = self.pool.tasks_list[0]
                    self.pool.tasks_list = self.pool.tasks_list[1:]
                # If this was the last task, signal that the pool is now idle.
                if task_count == 1:
                    self.pool.no_tasks.set()

            if current_task is not None:
                (current_device, script, location, neighbourhood) = current_task

                # Acquire the lock for the specific location, ensuring safe data access.
                with current_device.glocks[location]:
                    common_data = []
                    # Gather data from all devices in the neighborhood.
                    for neighbour in neighbourhood:
                        data = neighbour.get_data(location)
                        if data is not None:
                            common_data.append(data)
                    data = current_device.get_data(location)
                    if data is not None:
                        common_data.append(data)
                    
                    # Execute script and disseminate results.
                    if common_data != []:
                        result = script.run(common_data)
                        for neighbour in neighbourhood:
                            neighbour.set_data(location, result)
                        current_device.set_data(location, result)

class WorkPool(object):
    """
    A custom thread pool implementation using a list as a task queue,
    coordinated by a Lock, a Semaphore, and an Event.
    """
    def __init__(self, size):
        self.size = size
        self.tasks_list = []
        self.task_lock = Lock()       # Lock to protect the tasks_list.
        self.task_sign = Semaphore(0) # Semaphore to signal available tasks.
        self.no_tasks = Event()       # Event to signal when the task list is empty.
        self.no_tasks.set()
        self.stop = False             # Flag to signal workers to shut down.

        self.workers = []
        for i in xrange(self.size):
            worker = WorkerThread(i, self)
            self.workers.append(worker)

        for worker in self.workers:
            worker.start()

    def work(self, task):
        """Adds a task to the pool for a worker to execute."""
        with self.task_lock:
            self.tasks_list.append(task)
            self.task_sign.release() # Signal a waiting worker.
            if self.no_tasks.is_set():
                self.no_tasks.clear()

    def finish_work(self):
        """Blocks until the `no_tasks` event is set, indicating an idle pool."""
        self.no_tasks.wait()

    def end(self):
        """Shuts down the thread pool gracefully."""
        self.finish_work()
        self.stop = True
        # Release all workers from the semaphore so they can see the stop flag.
        for _ in self.workers:
            self.task_sign.release()
        for thread in self.workers:
            thread.join()
