"""
@83fbb450-9856-48f3-ab7b-242bcc75c940/device.py
@brief Simulation of a distributed sensor network with concurrent script execution.
Architecture: Peer-to-peer device model where individual nodes perform local data processing and share results with neighbors.
Functional Utility: Orchestrates multi-threaded data acquisition, script dispatching, and cross-device synchronization using barriers.
Synchronization: Employs threading.Event for signaling, threading.Lock for data consistency, and a reusable barrier for temporal alignment across the network.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
import multiprocessing
from queue import *

class Device(object):
    """
    @brief Core entity representing a physical or virtual sensor node.
    State: Maintains local sensor telemetry, assigned processing scripts, and synchronization primitives.
    Relationships: Interacts with a Supervisor for discovery and neighboring Devices for data exchange.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Numerical identifier for the node.
        @param sensor_data Dictionary mapping locations to local telemetry values.
        @param supervisor Management entity for network topology.
        """
        self.currentScript = 0
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals that a new processing batch is ready.
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.queue = Queue()
        self.hash = {} # Mapping: Location -> threading.Lock for atomic data access.


    def __str__(self):
        return "Device %d" % self.device_id

    def get_unique_id(self, devices):
        """
        @brief Heuristic to identify the highest ID in the set for designating a coordinator.
        """
        max_id = 0;
        for device in devices:
            if (device.device_id > max_id):
                max_id = device.device_id
        return max_id

    def setup_devices(self, devices):
        """
        @brief Network-wide initialization and synchronization setup.
        Logic: The node with the highest ID acts as the master to initialize shared locks and the global barrier.
        Invariant: All devices in the provided list must share the same barrier and lock registry.
        """
        if (self.device_id == self.get_unique_id(devices)):
            self.barrier = ReusableBarrierCond(len(devices)) 
            for device in devices:
                for k in device.sensor_data:
                    # Functional Utility: Per-location locks prevent race conditions during concurrent data reads/writes.
                    self.hash[k] = Lock()
            
            for device in devices:
                device.barrier = self.barrier
                device.hash = self.hash
        
        pass

    def assign_script(self, script, location):
        """
        @brief Queues a processing task for the next execution timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Sentinel value (None) triggers the start of processing.
            self.script_received.set() 


    def get_data(self, location):
        """
        @brief Thread-safe retrieval of sensor data.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Thread-safe update of sensor data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Finalizes device execution and joins the worker thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Lifecycle manager for device-level concurrency.
    Strategy: Spawns a pool of worker threads based on CPU count to execute scripts in parallel.
    """
    
    def worker(self, q):
        """
        @brief Execution kernel for a single script task.
        Synchronization: Acquires the location-specific lock before performing multi-device data aggregation.
        """
        while True:
            item = q.get()
            my_device = item[0]
            script = item[1]
            location = item[2]
            neighbours = item[3]
            
            # Termination: Sentinel check to stop the worker thread.
            if location is None:
                q.task_done()
                break
            

            # Block Logic: Critical section for distributed data processing.
            my_device.hash[location].acquire()
            script_data = []
            
            # Logic: Aggregates telemetry from the node and all its neighbors.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = my_device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Functional Utility: Executes the assigned algorithm on the aggregated data set.
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                my_device.set_data(location, result)
            
            my_device.hash[location].release()
            q.task_done()
            



    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.numCPUs = multiprocessing.cpu_count()


    def run(self):
        """
        @brief Orchestrates the temporal loop of the device.
        Flow: Discovery -> script dispatching -> parallel execution -> global synchronization.
        """
        q = Queue()
        threads = {}
        # Initialization: Scaffolds the internal thread pool.
        for i in range(self.numCPUs):
            threads[i] = Thread(target=self.worker, args =(q, ))
            threads[i].daemon = True
            threads[i].start()

        while True:
            # Logic: Periodically refreshes neighborhood topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Synchronization: Wait for the 'go' signal from the supervisor.
            self.device.script_received.wait()
            
            # Block Logic: Dispatches all queued scripts to the worker pool.
            for (script, location) in self.device.scripts:
                q.put((self.device, script, location, neighbours)) 

            q.join() # Invariant: All local script tasks must complete before the barrier.
            self.device.script_received.clear()
            
            # Barrier: Global synchronization point ensures all devices have finished the current timepoint.
            self.device.barrier.wait()
        
        # Shutdown Sequence: Signals all pool workers to terminate.
        for i in range(self.numCPUs):
            q.put((None, None, None, None))
        
        for i in range(self.numCPUs):
            threads[i].join()


# --- Implementation of basic Queue primitives ---

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
    @brief Multi-producer, multi-consumer FIFO queue implementation.
    Synchronization: Uses Condition variables (not_empty, not_full) for blocking I/O and all_tasks_done for joining.
    """
    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self._init(maxsize)
        
        self.mutex = _threading.Lock()
        
        self.not_empty = _threading.Condition(self.mutex)
        
        self.not_full = _threading.Condition(self.mutex)
        
        self.all_tasks_done = _threading.Condition(self.mutex)
        self.unfinished_tasks = 0

    def task_done(self):
        """
        @brief Signals that a previously enqueued task is complete.
        Logic: Atomically decrements the counter and notifies join() if the counter hits zero.
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
        @brief Blocks until all items in the queue have been processed.
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
        @brief Inserts an item into the queue.
        Synchronization: Blocks on not_full condition if the queue is saturated.
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
        return self.put(item, False)

    def get(self, block=True, timeout=None):
        """
        @brief Removes and returns an item from the queue.
        Synchronization: Blocks on not_empty condition if the queue is depleted.
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
    @brief Queue variant that retrieves items in sorted order (lowest priority first).
    Logic: Uses the heapq module to maintain a min-heap structure.
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
    @brief Last-In, First-Out (LIFO) queue variant.
    """
    def _init(self, maxsize):
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop()
