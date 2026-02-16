"""
This module implements a device simulation using a single, globally shared
Task Scheduler and a pool of worker threads.

The architecture features a master device (id 0) that creates and distributes
all shared resources, including a task scheduler. All devices then add tasks
to this central scheduler. However, the implementation has significant flaws,
most notably a busy-wait polling loop in the worker threads, which is highly
inefficient. The synchronization model is also unusual, as devices synchronize
at a barrier after task *submission*, not task *completion*.
"""

from threading import Thread, Lock, Event, Semaphore

class ReusableBarrierSem(object):
    """A correct, two-phase reusable barrier for thread synchronization."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        self.phase1(); self.phase2()

    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads): self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads): self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

def add_lock_for_location(lock_per_location, location):
    """Helper to add a new (location, Lock) tuple to a list."""
    lock_per_location.append((location, Lock()))

def get_lock_for_location(lock_per_location, location):
    """Helper to find a lock for a given location by linear scan."""
    for (loc, lock) in lock_per_location:
        if loc == location:
            return lock
    return None


class Device(object):
    """Represents a device that produces tasks for a central scheduler."""
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None
        self.taskscheduler = None
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def share_barrier(self, barrier):
        self.barrier = barrier

    def share_taskscheduler(self, taskscheduler):
        self.taskscheduler = taskscheduler

    def setup_devices(self, devices):
        """
        Master-driven setup (device 0) to create and distribute all shared
        resources, including locks, a barrier, and the task scheduler.
        """
        if self.device_id == 0:
            lock_per_location = []
            # Find all unique locations and create a lock for each.
            for device in devices:
                for location in device.sensor_data:
                    lock = get_lock_for_location(lock_per_location, location)
                    if lock is None:
                        add_lock_for_location(lock_per_location, location)

            self.barrier = ReusableBarrierSem(len(devices))
            self.taskscheduler = TaskScheduler(lock_per_location)

            # Distribute shared objects to all devices.
            for device in devices:
                device.share_taskscheduler(self.taskscheduler)
                device.share_barrier(self.barrier)

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data for a given location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Sets data for a given location."""
        self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It adds tasks to the central scheduler.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        # Inefficient busy-wait for setup to complete.
        while self.device.barrier is None:
            pass

        while True:
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # Master device is responsible for shutting down the scheduler.
                if self.device.device_id == 0:
                    self.device.taskscheduler.finish = True
                    self.device.taskscheduler.wait_workers()
                break

            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Add all assigned scripts as tasks to the central scheduler.
            for (script, location) in self.device.scripts:
                new_task = Task(self.device, script, location, neighbours)
                self.device.taskscheduler.add_task(new_task)

            # --- AMBIGUOUS SYNCHRONIZATION ---
            # The device waits at the barrier immediately after DISPATCHING its tasks,
            # not after they are COMPLETED. This can cause tasks from time step T
            # to still be running when devices proceed to setup time step T+1.
            self.device.barrier.wait()


class Task(object):
    """A simple data class representing a unit of work."""
    def __init__(self, device, script, location, neighbours):
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def execute(self):
        """The core logic of the task: gather, compute, and disseminate."""
        script_data = []
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)


class TaskScheduler(object):
    """A custom thread pool that manages a queue of tasks for worker threads."""
    def __init__(self, lock_per_location):
        self.nr_threads = 16
        self.lock_per_location = lock_per_location
        self.workpool = [] # A list used as a LIFO work queue.
        self.workpool_lock = Lock()
        self.workers_list = []
        self.finish = False # Flag to signal shutdown.
        self.start_workers()

    def add_task(self, new_task):
        """Safely adds a task to the work pool."""
        with self.workpool_lock:
            self.workpool.append(new_task)

    def get_task(self):
        """Safely retrieves a task from the work pool, returning None if empty."""
        self.workpool_lock.acquire()
        ret = self.workpool.pop() if self.workpool != [] else None
        self.workpool_lock.release()
        return ret

    def start_workers(self):
        """Creates and starts all worker threads."""
        tid = 0
        while tid < self.nr_threads:
            thread = Worker(self)
            self.workers_list.append(thread)
            tid += 1
        for worker in self.workers_list:
            worker.start()

    def wait_workers(self):
        """Waits for all worker threads to complete."""
        for worker in self.workers_list:
            worker.join()

    def get_lock_per_location(self, location):
        """Finds the lock for a given location via linear scan."""
        for (loc, lock) in self.lock_per_location:
            if loc == location:
                return lock
        return None


class Worker(Thread):
    """
    A worker thread that executes tasks from the TaskScheduler.
    """
    def __init__(self, taskscheduler):
        Thread.__init__(self)
        self.taskscheduler = taskscheduler

    def run(self):
        """
        Main loop: continuously polls for tasks and executes them. This polling
        is inefficient as it busy-waits when the queue is empty.
        """
        while True:
            if self.taskscheduler.finish == True:
                break

            while True:
                task = self.taskscheduler.get_task()
                if task is None:
                    break # Work queue is empty, re-check finish flag.
                
                # Acquire the correct lock for the task's location before executing.
                with self.taskscheduler.get_lock_per_location(task.location):
                    task.execute()
