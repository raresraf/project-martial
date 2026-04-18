
"""
@72110648-fc88-4120-a01f-5eb27de52e59/Queue.py
@brief Synchronized queuing primitives and threaded device simulation framework.
This module provides a dual-purpose implementation. The first part defines 
multi-producer, multi-consumer queue structures (FIFO, LIFO, and Priority) with 
rigorous condition-based synchronization. The second part implements a 
distributed device simulation where agents coordinate parallel script execution 
across spatial neighborhoods using the aforementioned queues and global 
temporal barriers.

Domain: Concurrent Programming, Data Structures, Distributed Simulation.
"""

from time import time as _time
try:
    import threading as _threading
except ImportError:
    import dummy_threading as _threading
from collections import deque
import heapq

__all__ = ['Empty', 'Full', 'Queue', 'PriorityQueue', 'LifoQueue']

class Empty(Exception):
    """
    Functional Utility: Exception raised by non-blocking get() or timed get() 
    when the queue is empty.
    """
    pass

class Full(Exception):
    """
    Functional Utility: Exception raised by non-blocking put() or timed put() 
    when the queue has reached maxsize.
    """
    pass

class Queue:
    """
    Functional Utility: Thread-safe FIFO (First-In, First-Out) queue.
    Logic: Uses a mutex lock to protect internal state and condition variables 
    (not_empty, not_full) to implement blocking semantics for producers and consumers. 
    It also tracks unfinished tasks to support join() synchronization.
    """
    
    def __init__(self, maxsize=0):
        """
        Constructor: Initializes the queue with an optional capacity limit.
        """
        self.maxsize = maxsize
        self._init(maxsize)
        
        # Synchronization: Core mutex for atomic state transitions.
        self.mutex = _threading.Lock()
        
        # Block Logic: Blocking condition for consumers.
        self.not_empty = _threading.Condition(self.mutex)
        
        # Block Logic: Blocking condition for producers.
        self.not_full = _threading.Condition(self.mutex)
        
        # Block Logic: Task completion rendezvous.
        self.all_tasks_done = _threading.Condition(self.mutex)
        self.unfinished_tasks = 0

    def task_done(self):
        """
        Functional Utility: Signals that a previously enqueued task is complete.
        Logic: Decrements the unfinished task counter and notifies any threads 
        waiting in join().
        """
        self.all_tasks_done.acquire()
        try:
            unfinished = self.unfinished_tasks - 1
            if unfinished <= 0:
                if unfinished < 0:
                    raise ValueError('task_done() called too many times')
                self.all_tasks_done.notify_all()
            self.unfinished_tasks = unfinished
        finally:
            self.all_tasks_done.release()

    def join(self):
        """
        Execution Logic: Blocks until all tasks have been processed (task_done called).
        """
        self.all_tasks_done.acquire()
        try:
            while self.unfinished_tasks:
                self.all_tasks_done.wait()
        finally:
            self.all_tasks_done.release()

    def qsize(self):
        """
        Functional Utility: Atomically retrieves the current number of items.
        """
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

    def empty(self):
        """
        Functional Utility: Atomically checks if the queue is depleted.
        """
        self.mutex.acquire()
        n = not self._qsize()
        self.mutex.release()
        return n

    def full(self):
        """
        Functional Utility: Atomically checks if the queue is at capacity.
        """
        self.mutex.acquire()
        n = 0 < self.maxsize == self._qsize()
        self.mutex.release()
        return n

    def put(self, item, block=True, timeout=None):
        """
        Execution Logic: Submits an item into the queue.
        Logic: Blocks on the not_full condition if capacity is reached. 
        Supports timed blocking and non-blocking modes.
        """
        self.not_full.acquire()
        try:
            if self.maxsize > 0:
                if not block:
                    if self._qsize() == self.maxsize:
                        raise Full
                elif timeout is None:
                    while self._qsize() == self.maxsize:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = _time() + timeout
                    while self._qsize() == self.maxsize:
                        remaining = endtime - _time()
                        if remaining <= 0.0:
                            raise Full
                        self.not_full.wait(remaining)
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()
        finally:
            self.not_full.release()

    def put_nowait(self, item):
        """
        Functional Utility: Non-blocking equivalent of put().
        """
        return self.put(item, False)

    def get(self, block=True, timeout=None):
        """
        Execution Logic: Retrieves an item from the queue.
        Logic: Blocks on the not_empty condition if the queue is empty.
        """
        self.not_empty.acquire()
        try:
            if not block:
                if not self._qsize():
                    raise Empty
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = _time() + timeout
                while not self._qsize():
                    remaining = endtime - _time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            item = self._get()
            self.not_full.notify()
            return item
        finally:
            self.not_empty.release()

    def get_nowait(self):
        """
        Functional Utility: Non-blocking equivalent of get().
        """
        return self.get(False)

    def _init(self, maxsize):
        self.queue = deque()

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.popleft()


class PriorityQueue(Queue):
    """
    Functional Utility: Thread-safe queue that retrieves items by priority.
    Logic: Uses a heap-based internal representation to ensure that the lowest 
    priority element is always retrieved first.
    """

    def _init(self, maxsize):
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item, heappush=heapq.heappush):
        heappush(self.queue, item)

    def _get(self, heappop=heapq.heappop):
        return heappop(self.queue)


class LifoQueue(Queue):
    """
    Functional Utility: Thread-safe LIFO (Last-In, First-Out) queue (Stack).
    """

    def _init(self, maxsize):
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop()

# --- Part 2: Device Simulation ---

from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierSem
import time
from Queue import *

class Device(object):
    """
    Functional Utility: Represent a cluster-aware simulated hardware unit.
    Logic: Orchestrates local sensor state and coordinates cluster-wide 
    synchronization. It uses a dynamic worker pool to execute assigned 
    scripts over neighborhood topologies.
    """
    
    devices_barrier = None
    nr_devices = 0

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Constructor: Initializes device identity, data storage, and the primary 
        execution thread.
        """
        Device.nr_devices += 1
        self.current_neighbours = None
        self.current_neighbours_set = False
        self.access_neighbours = Lock()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.workpool = Queue()
        self.timepoint_done = Event()
        self.threads = []
        self.passed = True
        self.passed_lock = Lock()
        self.device_barrier = None
        
        # Synchronization: Per-device thread pool barrier.
        self.threads_barrier = ReusableBarrierSem(1)
        self.start_barrier = ReusableBarrierSem(1)
        
        # Spawn initial worker.
        thread = DeviceThread(self, 1)
        self.threads.append(thread)
        thread.start()

    @classmethod
    def get_devices_barrier(cls):
        """
        Functional Utility: Static provider for the global simulation barrier.
        Logic: Ensures a singleton barrier instance for all devices in the cluster.
        """
        if cls.devices_barrier == None:
            cls.devices_barrier = ReusableBarrierSem(cls.nr_devices)
        return cls.devices_barrier

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Functional Utility: Cluster initialization protocol.
        """
        self.device_barrier = Device.get_devices_barrier()

    def assign_script(self, script, location):
        """
        Functional Utility: Dispatches computational tasks.
        Logic: Enqueues script into the workpool and dynamically scales the 
        thread pool up to a limit of 8 workers if workload increases.
        """
        if script is not None:
            self.workpool.put((script, location))
            self.scripts.append((script, location))

            # Block Logic: Dynamic worker scaling.
            if len(self.scripts) > len(self.threads) & len(self.threads) < 9:
                self.threads_barrier = ReusableBarrierSem(len(self.threads) + 1)
                thread = DeviceThread(self, len(self.threads) + 1)
                self.threads.append(thread)
                thread.start()
            self.script_received.set()

        else:
            # Block Logic: Termination signal.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Functional Utility: Atomic local data retrieval.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Functional Utility: Atomic local data update.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Functional Utility: Graceful termination sequence.
        """
        for thread in self.threads:
            thread.join()

class DeviceThread(Thread):
    """
    Functional Utility: Worker thread for parallel script execution.
    Logic: Iteratively pulls tasks from the device workpool. It performs 
    neighborhood data aggregation, executes the script, and propagates 
    results across neighbors before synchronizing at temporal barriers.
    """

    def __init__(self, device, name):
        """
        Constructor: Binds the worker to its parent device.
        """
        Thread.__init__(self)
        self.device = device
        self.name = "%d" % device.device_id + "%d" % name

    def run(self):
        """
        Execution Logic: Infinite simulation loop.
        """
        while True:
            
            # Block Logic: Temporal boundary logic.
            # Only the leader thread for each device retrieves new neighbor data.
            with self.device.passed_lock:
                if self.device.passed == True:
                    self.device.passed = False
                    self.device.current_neighbours = self.device.supervisor.get_neighbours()

            if self.device.current_neighbours is None:
                # Termination sentinel.
                break
            else:
                
                # Block Logic: Inner task consumption loop.
                self.device.script_received.wait()
                while True:
                    try:
                        (script, location) = self.device.workpool.get_nowait()
                    except Exception:
                        # Queue exhausted.
                        break

                    script_data = []
                    
                    # Block Logic: Data reconciliation across neighborhood.
                    try:
                        for device in self.device.current_neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                    except Exception:
                        break
                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        # Executes computational logic on aggregated inputs.
                        result = script.run(script_data)
                        
                        try:
                            for device in self.device.current_neighbours:
                                device.set_data(location, result)
                        except Exception:
                            break
                        
                        self.device.set_data(location, result)
            
            # Block Logic: Synchronization rendezvous.
            self.device.timepoint_done.wait()
            self.device.threads_barrier.wait()
            self.device.timepoint_done.clear()

            
            # Block Logic: Post-step cluster synchronization.
            with self.device.passed_lock:
                if self.device.passed != True:
                    self.device.device_barrier.wait()
                    self.device.passed = True
                    
                    # Refill workpool for next cycle.
                    for (script,location) in self.device.scripts:
                        self.device.workpool.put((script,location))
        
        # Block Logic: Shutdown protocols.
        self.device.threads_barrier.stop_barrier()
        self.device.devices_barrier.stop_barrier()
        self.device.timepoint_done.set()
