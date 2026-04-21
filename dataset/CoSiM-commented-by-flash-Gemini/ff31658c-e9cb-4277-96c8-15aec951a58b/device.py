
"""
@file device.py
@brief Distributed device simulation framework with multi-threaded task orchestration.

Functional Intent: Defines a Device entity that participates in a distributed network, 
capable of executing scripts based on sensor data gathered from its neighborhood. 
Manages parallel worker threads and leverages barrier synchronization to coordinate 
time-discrete processing steps across the swarm.

Domain: Production Systems, Distributed Computing, Parallel Processing.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
import multiprocessing
from queue import *

class Device(object):
    """
    @brief Represents a physical or virtual device in a distributed network.
    
    Functional Utility: Serves as the primary container for sensor data and 
    processing scripts. Coordinates with a supervisor to identify neighbors 
    and perform collective computations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device with unique ID and initial data state.
        """
        self.currentScript = 0
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Synchronization: Events and barriers for coordinating cross-device state transitions.
        self.script_received = Event()
        self.timepoint_done = Event()
        self.barrier = None
        
        # Logic: Spawns the management thread responsible for worker orchestration.
        self.thread = DeviceThread(self)
        self.thread.start()
        
        self.queue = Queue()
        self.hash = {} # Logic: Shared lock pool for location-based data access.


    def __str__(self):
        return "Device %d" % self.device_id

    def get_unique_id(self, devices):
        """
        @brief Helper for identifying the highest device ID in a cluster.
        """
        max_id = 0;
        for device in devices:
            if (device.device_id > max_id):
                max_id = device.device_id
        return max_id

    def setup_devices(self, devices):
        """
        @brief Orchestrates the cluster-wide setup of synchronization primitives.
        
        Algorithm: Elects the highest ID device to initialize shared locks and barriers.
        Invariant: All devices in the group will share the same ReusableBarrierCond instance.
        """
        if (self.device_id == self.get_unique_id(devices)):
            self.barrier = ReusableBarrierCond(len(devices)) 
            for device in devices:
                for k in device.sensor_data:
                    # Logic: Creates a mutex per data location to prevent concurrent write corruption.
                    self.hash[k] = Lock()
            
            # Side Effect: Propagates shared sync objects to all peers.
            for device in devices:
                device.barrier = self.barrier
                device.hash = self.hash
        
        pass

    def assign_script(self, script, location):
        """
        @brief Queues a script for execution on specific data locations.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signaling None indicates the end of a script batch for the current epoch.
            self.script_received.set() 


    def get_data(self, location):
        """
        @brief Thread-safe retrieval of sensor data (requires external lock acquisition).
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
        @brief Gracefully terminates the device orchestration thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Management thread that supervises a pool of parallel worker threads.
    
    Logic: Implements a producer-consumer pattern where tasks (scripts) are 
    dispatched to CPU-optimized worker threads.
    """
    
    def worker(self, q):
        """
        @brief Execution logic for individual worker threads.
        
        Algorithm: Data-gathering neighborhood reduction.
        Logic: 
        1. Acquires location-specific lock.
        2. Aggregates data from neighbors.
        3. Executes transformation script.
        4. Broadcasts results back to the neighborhood.
        """
        while True:
            item = q.get()
            my_device = item[0]
            script = item[1]
            location = item[2]
            neighbours = item[3]
            
            # Poison Pill: Sentinel value to terminate the worker thread.
            if location is None:
                q.task_done()
                break
            
            # Synchronization: Critical section for distributed data consistency.
            my_device.hash[location].acquire()
            script_data = []
            
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = my_device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Functional Intent: Perform user-defined data processing.
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
        @brief Core loop for the device manager.
        
        Logic: Coordinates task distribution across local CPU cores and waits 
        at a cluster barrier after each epoch to ensure global state convergence.
        """
        q = Queue()
        threads = {}
        # Optimization: Scales thread pool size to available hardware concurrency.
        for i in range(self.numCPUs):
            threads[i] = Thread(target=self.worker, args =(q, ))
            threads[i].daemon = True
            threads[i].start()

        while True:
            # Block Logic: Neighbor discovery via supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()
            
            # Task Dispatch: Populates the work queue for the parallel worker pool.
            for (script, location) in self.device.scripts:
                q.put((self.device, script, location, neighbours)) 

            # Synchronization: Blocks manager until all local scripts for this epoch are processed.
            q.join()            
            self.device.script_received.clear()
            
            # Synchronization: Global barrier to ensure all devices have completed the current step.
            self.device.barrier.wait()
        
        # Shutdown Sequence: Cleanly terminates worker threads.
        for i in range(self.numCPUs):
            q.put((None, None, None, None))
        
        for i in range(self.numCPUs):
            threads[i].join()


# Standard internal library implementations for concurrency primitives.
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
    @brief Thread-safe FIFO queue implementation with condition-based blocking.
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
        @brief Enqueues an item, potentially blocking if the queue is full.
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
    @brief Variant of Queue that retrieves items in priority order.
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
    @brief Variant of Queue that retrieves items in last-in, first-out order.
    """
    def _init(self, maxsize):
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop()
