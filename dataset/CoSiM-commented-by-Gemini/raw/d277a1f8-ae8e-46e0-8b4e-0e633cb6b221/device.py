"""
This module implements a distributed device simulation with a centralized
architecture for concurrency and task management.

Key architectural features:
- A single "master" device (the first in the list) is responsible for creating
  and distributing shared resources to all other devices.
- A global, shared thread pool (`WorkPool`) processes all computational tasks
  from all devices.
- A global, shared barrier (`ReusableBarrier`) synchronizes all devices at the
  end of each time step.
- A global, shared dictionary of locks (`glocks`) provides fine-grained,
  location-specific synchronization across the entire system.

Note: This script depends on local `pool.py` and `reusable_barrier.py` modules
and uses Python 2 syntax (e.g., `xrange`). The code for the pool is included
in this file, suggesting some refactoring may have occurred.
"""

from threading import Event, Thread, Lock
from pool import WorkPool
from reusable_barrier import ReusableBarrier

class Device(object):
    """
    Represents a single node in the simulation.

    Each device submits its computational tasks to a global thread pool and uses
    a shared barrier to synchronize with other devices at the end of each time step.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()
        self.scripts = []

        self.timepoint_done = Event()
        self.other_devs = []
        # This lock protects the device's own sensor_data dictionary.
        self.slock = Lock()

        # --- Shared objects, initialized in setup_devices ---
        self.barrier = None
        self.process = Event()
        self.global_thread_pool = None
        self.glocks = {}

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs centralized setup of shared resources.

        The first device in the `devices` list acts as a master, creating the
        global thread pool, barrier, and lock dictionary. All other devices
        receive a reference to these shared objects.

        :param devices: A list of all Device objects in the simulation.
        """
        self.other_devs = devices
        # Device 0 is designated as the master for initialization.
        if self.device_id == self.other_devs[0].device_id:
            locks = {}
            for loc in self.sensor_data:
                locks[loc] = Lock()
            dev_cnt = len(devices)
            self.glocks = locks
            self.barrier = ReusableBarrier(dev_cnt)
            self.global_thread_pool = WorkPool(16)
        else:
            # Other devices get references to the master's shared objects.
            for loc in self.sensor_data:
                self.other_devs[0].glocks[loc] = Lock()
            self.glocks = self.other_devs[0].glocks
            self.global_thread_pool = self.other_devs[0].global_thread_pool
            self.barrier = self.other_devs[0].barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be run in the current time step.

        :param script: The script object. A value of None signals that all
                       scripts for the time step have been assigned.
        :param location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        Thread-safely gets data from this device's sensor_data.
        """
        ret = None
        with self.slock:
            if location in self.sensor_data:
                ret = self.sensor_data[location]
        return ret

    def set_data(self, location, data):
        """
        Thread-safely sets data in this device's sensor_data.
        """
        with self.slock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating its lifecycle.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop, defining the device's behavior at each time step.

        The flow is:
        1. Wait for all scripts for the time step to be assigned.
        2. Add all scripts as tasks to the global thread pool.
        3. Wait for the global thread pool to complete all submitted tasks.
        4. Wait at a global barrier to synchronize with all other devices.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # On shutdown, the master device terminates the global pool.
                if self.device.device_id is self.device.other_devs[0].device_id:
                    self.device.global_thread_pool.end()
                break

            self.device.script_received.wait()

            # Add this device's work to the global thread pool.
            for (script, location) in self.device.scripts:
                self.device.global_thread_pool.work((self.device,
                									script,
                									location,
                									neighbours))

            # Wait for all tasks from all devices to be completed.
            self.device.global_thread_pool.finish_work()
            self.device.script_received.clear()
            # Synchronize with all other devices before the next time step.
            self.device.barrier.wait()


from threading import Lock, Event, Semaphore, Thread

class WorkerThread(Thread):
    """
    A worker thread managed by the WorkPool.
    """
    def __init__(self, i, parent_work_pool):
        Thread.__init__(self, name="WorkerThread%d" % i)
        self.pool = parent_work_pool

    def run(self):
        """
        The worker's main loop: fetch and execute tasks.
        """
        while True:
            self.pool.task_sign.acquire()  # Wait for a task to be available.
            if self.pool.stop:
                break
            
            current_task = (None, None, None, None)
            with self.pool.task_lock:
                task_count = len(self.pool.tasks_list)
                if task_count > 0:
                    current_task = self.pool.tasks_list.pop(0) # FIFO
                
                # If this was the last task, signal that the pool is idle.
                if task_count == 1:
                    self.pool.no_tasks.set()

            if current_task is not None and current_task[0] is not None:
                (current_device, script, location, neighbourhood) = current_task
                
                # Acquire the global lock for the specific data location.
                with current_device.glocks[location]:
                    common_data = []
                    
                    # Gather data from the target device and its neighbors.
                    for neighbour in neighbourhood:
                        data = neighbour.get_data(location)
                        if data is not None:
                            common_data.append(data)
                    
                    data = current_device.get_data(location)
                    if data is not None:
                        common_data.append(data)

                    if common_data:
                        # Run the script and broadcast the results.
                        result = script.run(common_data)
                        for neighbour in neighbourhood:
                            neighbour.set_data(location, result)
                        
                        current_device.set_data(location, result)

class WorkPool(object):
    """
    A simple, globally shared thread pool.
    """
    def __init__(self, size):
        self.size = size
        self.tasks_list = [] 
        self.task_lock = Lock()
        self.task_sign = Semaphore(0)
        self.no_tasks = Event()  # Signalled when the task list is empty.
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
        Adds a task to the pool's work queue.
        
        :param task: The task to be executed by a worker thread.
        """
        with self.task_lock:
            self.tasks_list.append(task)
            self.task_sign.release()  # Wake up one worker.
            if self.no_tasks.is_set():
                self.no_tasks.clear()

    def finish_work(self):
        """Blocks until all tasks in the pool have been processed."""
        self.no_tasks.wait()

    def end(self):
        """Shuts down the thread pool, terminating all worker threads."""
        self.finish_work()
        self.stop = True
        # Wake up all threads so they can check the stop flag and exit.
        for _ in self.workers:
            self.task_sign.release()
        for thread in self.workers:
            thread.join()
