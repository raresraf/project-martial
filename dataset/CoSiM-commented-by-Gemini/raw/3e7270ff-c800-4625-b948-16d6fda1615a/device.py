
"""
@brief A non-functional device simulation with a thread pool.
@file device.py

This module attempts to implement a distributed device simulation using a
thread pool for task execution. It appears to be a combination of several
different code snippets, including a self-contained `ThreadPool` and multiple,
conflicting `ReusableBarrier` implementations.

WARNING: SEVERE ARCHITECTURAL FLAWS - THIS CODE IS NON-FUNCTIONAL.
1.  **Broken State Management**: The `setup_devices` method is critically flawed.
    Instead of creating a single set of shared resources, *each device* creates
    its own `ReusableBarrier` and its own list of `locks`. This completely
    defeats the purpose of synchronization, as there is no actual shared state
    between devices.
2.  **Guaranteed Deadlock**: In the `DeviceThread.run` loop, the code attempts
    to have one thread wait on every other device's barrier (`for dev in ...
    dev.barrier.wait()`). Since each barrier is a separate object belonging to a
    different device (due to flaw #1), this will cause an immediate and
    unrecoverable deadlock.
3.  **Unsafe Locking**: In `executeScript`, locks are acquired and released
    manually and do not cover the entire critical section, creating race conditions.
4.  **Redundant Code**: The file defines `ReusableBarrier` multiple times and
    includes an unused `MyThread` class at the end, indicating a copy-paste origin.
"""

from threading import Event, Thread, Lock , Condition
# Note: The file defines its own ThreadPool, making this import potentially misleading.
from queue import Worker, ThreadPool
# Note: The file defines its own ReusableBarrier, making this import misleading.
from reusable_barrier_semaphore import ReusableBarrier

class Device(object):
    """Represents a device node in the broken simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.wait_neighbours = Event()
        self.scripts = []
        self.neighbours = []
        self.allDevices = []
        self.locks = [] # This will be a non-shared list of locks.
        self.pool = ThreadPool(8) # Each device gets its own thread pool.
        self.lock = Lock()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        FATALLY FLAWED setup method. It creates separate, non-shared resources
        for each device, making synchronization impossible.
        """
        self.allDevices = devices
        # Every device creates its OWN barrier. They will not synchronize.
        self.barrier = ReusableBarrier(len(devices))

        # Every device creates its OWN list of locks.
        for i in range(0, 50):
            self.locks.append(Lock())
        pass

    def assign_script(self, script, location):
        """Adds a script to a list and also submits it to the thread pool."""
        if script is not None:
            self.scripts.append((script, location))
            self.pool.add_task(self.executeScript,script,location)
        else:
            self.script_received.set()

    def get_data(self, location):
        return self.sensor_data.get(location, None)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

    def executeScript(self,script,location):
        """
        The task function for the thread pool workers. The locking logic is
        flawed because the lock objects are not shared between devices.
        """
        self.wait_neighbours.wait()
        script_data = []

        # This logic attempts to lock on other devices' unshared lock objects.
        if self.neighbours:
            for device in self.neighbours:
                device.locks[location].acquire()
                data = device.get_data(location)
                device.locks[location].release()

                if data is not None:
                    script_data.append(data)
        
        # Manually and unsafely locking a non-shared lock.
        self.locks[location].acquire()
        data = self.get_data(location)
        self.locks[location].release()

        if data is not None:
            script_data.append(data)

        if script_data:
            result = script.run(script_data)
            if self.neighbours:
                for device in self.neighbours:
                    device.locks[location].acquire()
                    device.set_data(location, result)
                    device.locks[location].release()

            self.locks[location].acquire()
            self.set_data(location, result)
            self.locks[location].release()


class DeviceThread(Thread):
    """The main control thread for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main loop that will inevitably deadlock.
        """
        while True:
            self.device.script_received.clear()
            self.device.wait_neighbours.clear()

            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.wait_neighbours.set()

            if self.device.neighbours is None:
                # Attempt a graceful shutdown of the pool.
                self.device.pool.wait_completion()
                self.device.pool.terminateWorkers()
                self.device.pool.threadJoin()
                return
            
            # Redundantly re-adds tasks that were already added in assign_script.
            for (script, location) in self.device.scripts:
                self.device.pool.add_task(self.device.executeScript,script,location)

            self.device.script_received.wait()
            self.device.pool.wait_completion()

            # --- DEADLOCK ---
            # This thread will now try to wait on barriers belonging to other
            # threads, which will never complete.
            for dev in self.device.allDevices:
                dev.barrier.wait()


# --- Start of self-contained ThreadPool implementation ---
from Queue import Queue
from threading import Thread

class Worker(Thread):
    """A worker thread that consumes tasks from a queue."""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            # A `None` function is the signal to terminate.
            if func is None:
                self.tasks.task_done()
                break
            try:
                func(*args, **kargs)
            except Exception as e:
                print(e)
            self.tasks.task_done()


class ThreadPool:
    """A simple thread pool implementation."""
    def __init__(self, num_threads):
        self.tasks = Queue(99999)
        self.workers = []
        for _ in range(num_threads):
            self.workers.append(Worker(self.tasks))

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue."""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Block until all tasks in the queue are processed."""
        self.tasks.join()

    def terminateWorkers(self):
        """Signal all workers to terminate."""
        for _ in self.workers:
            self.tasks.put((None, (), {}))

    def threadJoin(self):
        """Join all worker threads."""
        for worker in self.workers:
            worker.join()
# --- End of self-contained ThreadPool implementation ---


# --- Start of a redundant, flawed, and unused ReusableBarrier/MyThread ---
from threading import *

class ReusableBarrier():
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
                                                 

class MyThread(Thread):
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",
# --- End of redundant/unused code ---