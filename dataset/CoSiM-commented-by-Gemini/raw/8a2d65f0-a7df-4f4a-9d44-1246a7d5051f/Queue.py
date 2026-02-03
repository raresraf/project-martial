"""
This module provides a thread-safe queue implementation and a simulated device
for a distributed sensor network.

The `Queue`, `PriorityQueue`, and `LifoQueue` classes are standard data
structures for managing work in a multi-threaded environment. The `Device`
class simulates a node in a sensor network that processes data using scripts
coordinated by a supervisor.
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
    A thread-safe, first-in, first-out (FIFO) queue.

    This class provides a classic queue implementation with support for
    blocking operations, timeouts, and tracking of unfinished tasks.
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
        Indicates that a formerly enqueued task is complete.

        Used by queue consumer threads. For each get() used to fetch a task,
        a subsequent call to task_done() tells the queue that the processing
        on the task is complete.
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
        Blocks until all items in the queue have been gotten and processed.
        """
        self.all_tasks_done.acquire()
        try:
            # Invariant: The loop continues as long as there are unfinished tasks.
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
        Put an item into the queue.

        If optional args `block` is true and `timeout` is None (the default),
        block if necessary until a free slot is available. If `timeout` is
        a non-negative number, it blocks at most `timeout` seconds and raises
        the Full exception if no free slot was available within that time.
        Otherwise (`block` is false), put an item on the queue if a free slot
        is immediately available, else raise the Full exception.
        """
        self.not_full.acquire()
        try:
            if self.maxsize > 0:
                # Pre-condition: If the queue is full, wait for a slot to become
                # available.
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
        Remove and return an item from the queue.

        If optional args `block` is true and `timeout` is None (the default),
        block if necessary until an item is available. If `timeout` is a
        non-negative number, it blocks at most `timeout` seconds and raises
        the Empty exception if no item was available within that time.
        Otherwise (`block` is false), return an item if one is immediately
        available, else raise the Empty exception.
        """
        self.not_empty.acquire()
        try:
            # Pre-condition: If the queue is empty, wait for an item to be put.
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
    A variant of `Queue` that retrieves entries in priority order (lowest first).
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
    A variant of `Queue` that retrieves most recently added entries first.
    """

    def _init(self, maxsize):
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop()

from Queue import LifoQueue
from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

class Device(object):
    """
    Represents a single device in the simulated network.

    This class manages the device's state, including its sensor data and
    assigned scripts. It uses a pool of worker threads to execute scripts
    concurrently.
    """
    barry = None
    barry_is_set = Event()
    modify_value = {}
    location_locks = []


    def __init__(self, device_id, sensor_data, supervisor):
        
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
        self.my_data_lock = Lock()
        self.queue_lock = Lock()
        self.threads = []
        self.signal_end = False
        self.index = 0
        
        # Creates a pool of 8 worker threads.
        for i in range(8):
            self.threads.append(DeviceThread(self))
            self.threads[i].start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        self.devices = devices
        # Pre-condition: Device 0 initializes the shared barrier.
        if self.device_id == 0:
            Device.barry = ReusableBarrierCond(len(devices) * 8)
            Device.barry_is_set.set()

    def assign_script(self, script, location):
        
        
        if location not in Device.modify_value:
            Device.modify_value[location] = Lock()

        
        if script is None:
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
        
        for i in range(8):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread for a device.

    Each device has a pool of these threads, which continuously wait for
    scripts to be assigned and then execute them.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        Device.barry_is_set.wait()
        while True:
            Device.barry.wait()
            
            self.device.neighbours_lock.acquire()
            if self.device.signal_end:
                self.device.neighbours_lock.release()
                break
            if self.device.neighbours is None:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                if self.device.neighbours is None:
                    self.device.signal_end = True
                    self.device.neighbours_lock.release()
                    break
            self.device.neighbours_lock.release()
            # Invariant: The loop processes scripts from the device's queue until
            # a `None` script is encountered, which signals the end of the
            # timepoint.
            while True:
                self.device.queue_lock.acquire()
                
                if self.device.index < self.device.crt_scripts.qsize():
                    script, location = self.device.crt_scripts.queue[self.device.index]
                    if script is None:
                        self.device.crt_scripts.get()
                        self.device.queue_lock.release()
                        break
                    else:
                        
                        Device.modify_value[location].acquire()
                        self.device.index += 1
                        self.device.queue_lock.release()
                        script_data = []
                        
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
            
            Device.barry.wait()
            self.device.neighbours_lock.acquire()
            
            if self.device.neighbours is not None:
                self.device.neighbours = None
            if self.device.timepoint_done.is_set():
                self.device.timepoint_done.clear()
            self.device.neighbours_lock.release()
            self.device.queue_lock.acquire()
            if self.device.index > 0:
                self.device.index = 0
            self.device.queue_lock.release()
            
            Device.barry.wait()