"""
@a54dc587-4f6a-480a-99de-5e043e64abac/Queue.py
@brief Distributed sensor network simulation featuring multi-stage barrier synchronization and a custom thread-safe queue.
Architecture: Master-slave device model where node 0 initializes global synchronization primitives (Static Barrier, Event signaling).
Functional Utility: Orchestrates a parallel processing pipeline for sensor data, leveraging a LIFO queue for task dispatching and neighborhood-aware data aggregation.
Synchronization: Employs a complex multi-pass barrier pattern (wait-execute-wait-reset-wait) to ensure temporal consistency across the asynchronous network.
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
    @brief Multi-producer, multi-consumer FIFO queue implementation with blocking support.
    Synchronization: Uses a single mutex and three Condition variables (not_empty, not_full, all_tasks_done) for fine-grained thread coordination.
    """
    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self._init(maxsize)
        
        self.mutex = _threading.Lock() # Serializes access to internal storage and counters.
        
        self.not_empty = _threading.Condition(self.mutex) # Barrier for consumer threads when queue is drained.
        
        self.not_full = _threading.Condition(self.mutex) # Barrier for producer threads when queue is saturated.
        
        self.all_tasks_done = _threading.Condition(self.mutex) # Finalization barrier for joining threads.
        self.unfinished_tasks = 0

    def task_done(self):
        """
        @brief Signals the completion of an item processing task.
        Logic: Decrements global counter and notifies all_tasks_done wait-set if the queue is fully processed.
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
        @brief Blocks the calling thread until all enqueued items are processed.
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
        @brief Enqueues an item, with optional blocking and timeout.
        Synchronization: Blocks on not_full if the queue has reached maxsize.
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
            # Signal: Notifies a waiting consumer that data is available.
            self.not_empty.notify()
        finally:
            self.not_full.release()

    def put_nowait(self, item):
        return self.put(item, False)

    def get(self, block=True, timeout=None):
        """
        @brief Dequeues an item, with optional blocking and timeout.
        Synchronization: Blocks on not_empty if the queue is currently empty.
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
            # Signal: Notifies a waiting producer that a slot has been vacated.
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
    @brief Specialized queue using a min-heap to retrieve items in priority order.
    Logic: Leverages the heapq module for O(log N) insertion and extraction.
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
    @brief LIFO (Last-In, First-Out) queue variant; effectively a thread-safe stack.
    """
    def _init(self, maxsize):
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop()

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

class Device(object):
    """
    @brief core entity representing a sensor node with neighborhood awareness and parallel script execution.
    State: Class-level static barrier (barry) ensures all threads across all devices synchronize at temporal boundaries.
    Synchronization: Uses a hierarchy of locks (my_data_lock, queue_lock, modify_value) to serialize access to telemetry and task queues.
    """
    
    barry = None
    barry_is_set = Event() # Signals that the static barrier has been initialized by the master node.
    modify_value = {} # Mapping: Location -> Lock for cross-node data write consistency.
    location_locks = []


    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Numerical identifier (0 indicates the Master node).
        @param sensor_data Local telemetry store.
        @param supervisor Management interface for topology resolution.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbours = None
        self.scripts = []
        self.timepoint_done = Event()
        self.data_lock = Lock()
        self.neighbours_lock = Lock()
        self.crt_scripts = LifoQueue()
        self.devices = []
        self.ready = Event()
        self.my_data_lock = Lock() # Serializes reads/writes to this node's sensor_data.
        self.queue_lock = Lock() # Protects internal iteration state (self.index).
        self.threads = []
        self.signal_end = False # Shutdown flag for internal worker threads.
        self.index = 0
        
        # Initialization: Spawns 8 concurrent worker threads per device node.
        for i in range(8):
            self.threads.append(DeviceThread(self))
            self.threads[i].start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization setup.
        Logic: Node 0 acts as the Master to initialize the static barrier spanning all 8 threads of all devices.
        """
        self.devices = devices
        if self.device_id == 0:
            Device.barry = ReusableBarrierCond(len(devices) * 8)
            Device.barry_is_set.set()

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing task for the current timepoint.
        """
        if location not in Device.modify_value:
            Device.modify_value[location] = Lock()

        # Block Logic: Task dispatching.
        if script is None:
            # Sentinel: Triggers phase transition. Signals all 8 workers to finish.
            self.timepoint_done.set()
            for i in range(8):
                self.crt_scripts.put((None, None))
        else:
            self.crt_scripts.put((script, location))

    def get_data(self, location):
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Graceful teardown of the device thread pool.
        """
        for i in range(8):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    @brief Worker kernel for processing sensor data in parallel.
    Strategy: Operates in a triple-barrier loop to synchronize across neighborhood discovery, execution, and state reset phases.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core execution loop for the parallel worker.
        Flow: Barrier1 (Start Phase) -> Aggregation -> Computation -> Barrier2 (End Phase) -> Reset -> Barrier3 (Cleanup).
        """
        Device.barry_is_set.wait()
        while True:
            # Barrier 1: Start of current simulation timepoint.
            Device.barry.wait()
            
            self.device.neighbours_lock.acquire()
            if self.device.signal_end:
                self.device.neighbours_lock.release()
                break
            # Logic: Lazy neighborhood discovery for the current cycle.
            if self.device.neighbours is None:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                if self.device.neighbours is None:
                    self.device.signal_end = True
                    self.device.neighbours_lock.release()
                    break
            self.device.neighbours_lock.release()

            # Execution Phase: Processes all scripts currently assigned to the node.
            while True:
                self.device.queue_lock.acquire()
                
                # Block Logic: Task iteration.
                if self.device.index < self.device.crt_scripts.qsize():
                    script, location = self.device.crt_scripts.queue[self.device.index]
                    if script is None:
                        self.device.crt_scripts.get()
                        self.device.queue_lock.release()
                        break
                    else:
                        # Invariant: Aggregation must be atomic across nodes sharing a sensor location.
                        Device.modify_value[location].acquire()
                        self.device.index += 1
                        self.device.queue_lock.release()
                        script_data = []
                        
                        # Logic: Collects telemetry from local and neighbor nodes.
                        self.device.my_data_lock.acquire()
                        data = self.device.get_data(location)
                        self.device.my_data_lock.release()
                        if data is not None:
                            script_data.append(data)
                        for device in self.device.neighbours:
                            device.my_data_lock.acquire()
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                            device.my_data_lock.release()

                        if len(script_data) > 0:
                            # Functional Utility: Executes processing algorithm and propagates results.
                            result = script.run(script_data)
                            if result:
                                for dev in self.device.neighbours:
                                    dev.my_data_lock.acquire()
                                    dev.set_data(location, result)
                                    dev.my_data_lock.release()
                                self.device.my_data_lock.acquire()
                                self.device.set_data(location, result)
                                self.device.my_data_lock.release()
                        Device.modify_value[location].release()
                else:
                    self.device.queue_lock.release()
            
            # Barrier 2: Verification phase. Ensures all threads have finished local computations.
            Device.barry.wait()
            self.device.neighbours_lock.acquire()
            
            # State Reset: Clears iteration state and neighbors in preparation for the next timepoint.
            if self.device.neighbours is not None:
                self.device.neighbours = None
            if self.device.timepoint_done.is_set():
                self.device.timepoint_done.clear()
            self.device.neighbours_lock.release()
            self.device.queue_lock.acquire()
            if self.device.index > 0:
                self.device.index = 0
            self.device.queue_lock.release()
            
            # Barrier 3: Temporal alignment. Prevents any thread from entering T+1 until all threads reach this point.
            Device.barry.wait()
