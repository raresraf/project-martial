"""
@8675d965-59ac-4392-a117-a95cacab2d64/device.py
@brief Distributed sensor network simulation with multi-threaded script execution and neighborhood synchronization.
Architecture: Decentralized peer model where nodes process local/neighbor telemetry and synchronize via global barriers.
Functional Utility: Orchestrates high-concurrency sensor data aggregation, algorithmic processing, and result propagation.
Synchronization: Employs threading.Event for task signaling, threading.Lock for atomic data updates, and a reusable barrier for temporal alignment across the network.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
import multiprocessing
from queue import *

class Device(object):
    """
    @brief Functional entity representing a network-connected sensor node.
    State: Manages local telemetry store, assigned processing tasks, and discovery metadata.
    Relationships: Coordinates with a Supervisor for topological updates and adjacent Devices for data exchange.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier for the node instance.
        @param sensor_data Internal telemetry registry (Location -> Value).
        @param supervisor Orchestration entity for peer discovery.
        """
        self.currentScript = 0
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Synchronization: Signals the arrival of a new processing batch.
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.queue = Queue()
        self.hash = {} # Mapping: Location -> threading.Lock for serializing access to shared sensor points.


    def __str__(self):
        return "Device %d" % self.device_id

    def get_unique_id(self, devices):
        """
        @brief Heuristic to determine the network coordinator (highest ID).
        """
        max_id = 0;
        for device in devices:
            if (device.device_id > max_id):
                max_id = device.device_id
        return max_id

    def setup_devices(self, devices):
        """
        @brief Global network initialization and synchronization setup.
        Logic: The coordinator (max ID) initializes the shared barrier and the global lock registry for all nodes.
        Invariant: All participating devices must operate on the same barrier instance and lock set.
        """
        if (self.device_id == self.get_unique_id(devices)):
            self.barrier = ReusableBarrierCond(len(devices)) 
            for device in devices:
                for k in device.sensor_data:
                    # Functional Utility: Protects against race conditions during neighborhood data aggregation.
                    self.hash[k] = Lock()
            
            for device in devices:
                device.barrier = self.barrier
                device.hash = self.hash
        
        pass

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing algorithm for execution at the next timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Sentinel trigger to initiate the processing cycle.
            self.script_received.set() 


    def get_data(self, location):
        """
        @brief Thread-safe telemetry retrieval.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Thread-safe telemetry update.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Executes graceful termination of device operations.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Execution context manager for a single Device.
    Strategy: Spawns a pool of background worker threads to parallelize script execution across local CPU cores.
    """
    
    def worker(self, q):
        """
        @brief Kernel routine for processing a single sensor-location script.
        Synchronization: Locks the target location to ensure atomic aggregation across neighbor nodes.
        """
        while True:
            item = q.get()
            my_device = item[0]
            script = item[1]
            location = item[2]
            neighbours = item[3]
            
            # Termination: Sentinel check for thread pool shutdown.
            if location is None:
                q.task_done()
                break
            

            # Block Logic: Critical section for distributed data processing.
            my_device.hash[location].acquire()
            script_data = []
            
            # Logic: Gathers telemetry from the neighborhood and the local node.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = my_device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Functional Utility: Runs the script algorithm and propagates the result back to neighbors.
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
        @brief Main loop orchestrating the temporal progression of the device simulation.
        Flow: Discovery -> script dispatch -> parallel execution -> network-wide synchronization.
        """
        q = Queue()
        threads = {}
        # Initialization: Scaffolds the parallel worker pool.
        for i in range(self.numCPUs):
            threads[i] = Thread(target=self.worker, args =(q, ))
            threads[i].daemon = True
            threads[i].start()

        while True:
            # Logic: Refresh network neighborhood for the current cycle.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Synchronization: Wait for the global start signal.
            self.device.script_received.wait()
            
            # Block Logic: Offloads queued scripts to the worker pool for parallel completion.
            for (script, location) in self.device.scripts:
                q.put((self.device, script, location, neighbours)) 

            q.join() # Invariant: Local tasks must finalize before the device can reach the barrier.
            self.device.script_received.clear()
            
            # Barrier: Temporal alignment ensures no device proceeds to T+1 before all finish T.
            self.device.barrier.wait()
        
        # Shutdown: Signals pool workers to exit.
        for i in range(self.numCPUs):
            q.put((None, None, None, None))
        
        for i in range(self.numCPUs):
            threads[i].join()


# --- Concurrent Queue Implementation ---

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
    @brief Multi-threaded FIFO queue with blocking support.
    Synchronization: Uses Condition variables (not_empty, not_full) to manage producer-consumer coordination.
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
        @brief Signals completion of an enqueued task.
        Logic: Decrements active task counter and notifies join() if the queue is drained.
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
        @brief Blocking wait until all enqueued items are processed.
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
    @brief Min-heap based priority queue.
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
    @brief LIFO (Stack) queue implementation.
    """
    def _init(self, maxsize):
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop()
