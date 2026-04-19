"""
@b999a9e7-e272-46bb-a98f-7a8d6c077e3e/Queue.py
@brief Distributed sensor network simulation with dynamic thread-pool scaling and multi-level barrier synchronization.
Architecture: Master-less peer-to-peer model where each Device manages a local workpool and coordinates with neighbors.
Functional Utility: Orchestrates asynchronous script execution on aggregated telemetry data, featuring a custom thread-safe Queue implementation.
Synchronization: Employs a complex dual-barrier system (local threads_barrier and global device_barrier) and threading.Event for cross-thread signaling.
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
    "Exception raised by Queue.get(block=0)/get_nowait()."
    pass

class Full(Exception):
    "Exception raised by Queue.put(block=0)/put_nowait()."
    pass

class Queue:
    """
    @brief Multi-producer, multi-consumer FIFO queue implementation with condition-variable synchronization.
    State Management: Maintains internal item storage and tracks unfinished tasks for joining.
    Synchronization: Uses distinct conditions (not_empty, not_full, all_tasks_done) to manage thread contention and flow control.
    """
    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self._init(maxsize)
        
        self.mutex = _threading.Lock() # Protects internal data structures and counters.
        
        self.not_empty = _threading.Condition(self.mutex) # Consumer barrier for empty queue.
        
        self.not_full = _threading.Condition(self.mutex) # Producer barrier for full queue.
        
        self.all_tasks_done = _threading.Condition(self.mutex) # Finalization barrier.
        self.unfinished_tasks = 0

    def task_done(self):
        """
        @brief Signals the completion of an item processing task.
        Logic: Decrements global counter and notifies all_tasks_done wait-set if the queue is drained.
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
        @brief Blocks until all enqueued items have been processed and marked as done.
        """
        self.all_tasks_done.acquire()
        try:
            while self.unfinished_tasks:
                self.all_tasks_done.wait()
        finally:
            self.all_tasks_done.release()

    def qsize(self):
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

    def empty(self):
        self.mutex.acquire()
        n = not self._qsize()
        self.mutex.release()
        return n

    def full(self):
        self.mutex.acquire()
        n = 0 < self.maxsize == self._qsize()
        self.mutex.release()
        return n

    def put(self, item, block=True, timeout=None):
        """
        @brief Enqueues an item, potentially blocking if the queue is saturated.
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
            self.not_empty.notify() # Wake up waiting consumers.
        finally:
            self.not_full.release()

    def put_nowait(self, item):
        return self.put(item, False)

    def get(self, block=True, timeout=None):
        """
        @brief Dequeues an item, potentially blocking if the queue is empty.
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
            self.not_full.notify() # Wake up waiting producers.
            return item
        finally:
            self.not_empty.release()

    def get_nowait(self):
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
    @brief min-heap based priority queue.
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
    @brief Stack-like LIFO implementation.
    """
    def _init(self, maxsize):
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop()

# >>>> file: device.py

from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierSem
import time
from Queue import *

class Device(object):
    """
    @brief Core sensor node implementation with autonomous task processing.
    State: Maintains a dynamic worker pool that scales based on script density.
    Synchronization: uses a class-level static barrier (devices_barrier) to synchronize all network nodes at timepoint boundaries.
    """
    
    devices_barrier = None
    nr_devices = 0

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique numerical identifier.
        @param sensor_data Internal telemetry store (Location -> Value).
        @param supervisor Management interface for topological discovery.
        """
        Device.nr_devices += 1
        self.current_neighbours = None
        self.current_neighbours_set = False
        self.access_neighbours = Lock()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Synchronization: Signals to workers that new tasks are enqueued.
        self.scripts = []
        self.workpool = Queue()
        self.timepoint_done = Event() # Synchronization: Signals end of current processing cycle.
        self.threads = []
        self.passed = True
        self.passed_lock = Lock() # Serializes state transition between timepoints.
        self.device_barrier = None
        self.threads_barrier = ReusableBarrierSem(1)
        self.start_barrier = ReusableBarrierSem(1)
        
        # Initialization: Bootstraps the first worker thread.
        thread = DeviceThread(self, 1)
        self.threads.append(thread)
        thread.start()

    @classmethod
    def get_devices_barrier(cls):
        """
        @brief Singleton factory for the global device synchronization barrier.
        """
        if cls.devices_barrier == None:
            cls.devices_barrier = ReusableBarrierSem(cls.nr_devices)
        return cls.devices_barrier

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Bootstraps the global barrier for all nodes.
        """
        self.device_barrier = Device.get_devices_barrier()

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing algorithm for execution.
        Optimization: Predictively spawns additional worker threads (up to 8) if the workload exceeds current pool capacity.
        """
        if script is not None:
            self.workpool.put((script, location))
            self.scripts.append((script, location))

            # Block Logic: Dynamic scaling of the local thread pool.
            if len(self.scripts) > len(self.threads) & len(self.threads) < 9:
                self.threads_barrier = ReusableBarrierSem(len(self.threads) + 1)
                thread = DeviceThread(self, len(self.threads) + 1)
                self.threads.append(thread)
                thread.start()
            self.script_received.set()

        else:
            # Logic: Sentinel value triggers the finalization phase of the timepoint.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Graceful teardown of the worker thread pool.
        """
        for thread in self.threads:
            thread.join()

class DeviceThread(Thread):
    """
    @brief Parallel worker kernel for sensor data processing.
    Strategy: Implements a double-barrier loop to synchronize across local threads and global device nodes.
    """

    def __init__(self, device, name):
        Thread.__init__(self)
        self.device = device
        self.name = "%d" % device.device_id + "%d" % name

    def run(self):
        """
        @brief Core execution loop for the parallel task consumer.
        Flow: Topology discovery -> Script execution -> Local Barrier -> Global Barrier.
        """
        while True:
            # Block Logic: One thread per device performs topological lookup for the timepoint.
            with self.device.passed_lock:
                if self.device.passed == True:
                    self.device.passed = False
                    self.device.current_neighbours = self.device.supervisor.get_neighbours()

            if self.device.current_neighbours is None:
                # Termination: Break signal received from supervisor.
                break
            else:
                # Synchronization: Wait for task dispatch.
                self.device.script_received.wait()
                while True:
                    try:
                        # Logic: Non-blocking task acquisition to prevent worker hang during transition.
                        (script, location) = self.device.workpool.get_nowait()
                    except Exception:
                        # Termination: Internal workpool drained for this cycle.
                        break
                    
                    script_data = []
                    
                    # Logic: Data aggregation pass across neighborhood.
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
                        # Functional Utility: Executes processing logic and propagates results.
                        result = script.run(script_data)
                        
                        try:
                            for device in self.device.current_neighbours:
                                device.set_data(location, result)
                        except Exception:
                            break
                        
                        self.device.set_data(location, result)
            
            # Local Barrier: Ensures all local worker threads have completed their tasks.
            self.device.timepoint_done.wait()
            self.device.threads_barrier.wait()
            self.device.timepoint_done.clear()

            # Global Barrier: Temporal alignment across all network nodes.
            with self.device.passed_lock:
                if self.device.passed != True:
                    self.device.device_barrier.wait()
                    self.device.passed = True
                    
                    # State Reset: Re-fills workpool from persistent script list for the next timepoint.
                    for (script,location) in self.device.scripts:
                        self.device.workpool.put((script,location))
        
        # Shutdown Sequence: Signals all barriers to release blocked threads.
        self.device.threads_barrier.stop_barrier()
        self.device.devices_barrier.stop_barrier()
        self.device.timepoint_done.set()
