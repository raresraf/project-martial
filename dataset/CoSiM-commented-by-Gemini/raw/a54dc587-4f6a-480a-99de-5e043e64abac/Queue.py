"""
This file contains implementations for two distinct components:
1. A thread-safe queue library, nearly identical to Python's standard `Queue`,
   providing FIFO, LIFO, and Priority Queue implementations.
2. A complex device simulation framework that models a network of synchronized
   devices processing sensor data.

NOTE: These two components appear to have been concatenated into a single file.
The documentation addresses them as separate parts.
"""


# =============================================================================
# Part 1: Thread-Safe Queue Implementation
# =============================================================================

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
    A thread-safe, FIFO (First-In, First-Out) queue.

    This class is a reimplementation of Python's standard `Queue`. It allows
    multiple threads to safely add (`put`) and remove (`get`) items.
    """
    def __init__(self, maxsize=0):
        """Initializes a new Queue instance."""
        self.maxsize = maxsize
        self._init(maxsize)
        self.mutex = _threading.Lock()
        self.not_empty = _threading.Condition(self.mutex)
        self.not_full = _threading.Condition(self.mutex)
        self.all_tasks_done = _threading.Condition(self.mutex)
        self.unfinished_tasks = 0

    def task_done(self):
        """Indicate that a formerly enqueued task is complete."""
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
        """Blocks until all items in the queue have been gotten and processed."""
        self.all_tasks_done.acquire()
        try:
            while self.unfinished_tasks:
                self.all_tasks_done.wait()
        finally:
            self.all_tasks_done.release()

    def qsize(self):
        """Return the approximate size of the queue."""
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        self.mutex.acquire()
        n = not self._qsize()
        self.mutex.release()
        return n

    def full(self):
        """Return True if the queue is full, False otherwise."""
        self.mutex.acquire()
        n = 0 < self.maxsize == self._qsize()
        self.mutex.release()
        return n

    def put(self, item, block=True, timeout=None):
        """Put an item into the queue."""
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
        """Equivalent to put(item, False)."""
        return self.put(item, False)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue."""
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
        """Equivalent to get(False)."""
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
    """A thread-safe priority queue (min-heap)."""
    def _init(self, maxsize):
        self.queue = []

    def _put(self, item, heappush=heapq.heappush):
        heappush(self.queue, item)

    def _get(self, heappop=heapq.heappop):
        return heappop(self.queue)


class LifoQueue(Queue):
    """A thread-safe LIFO (Last-In, First-Out) queue, also known as a Stack."""
    def _init(self, maxsize):
        self.queue = []

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop()


# =============================================================================
# Part 2: Device Simulation Framework
# =============================================================================

from threading import Event, Thread, Lock
# The following are placeholders as the original modules are not available.
class ReusableBarrierCond:
    """A placeholder for a custom ReusableBarrier with Condition Variables."""
    def __init__(self, num_threads): pass
    def wait(self): pass

class Device(object):
    """
    Represents a node in a simulated, synchronized sensor network.

    This class uses a complex synchronization model based on shared, class-level
    state. All threads from all device instances synchronize on a global barrier (`barry`).
    Data access is coordinated through a dictionary of per-location locks (`modify_value`).
    Each device spawns a fixed number of worker threads to process scripts.
    """
    # Class-level state for global synchronization.
    barry = None
    barry_is_set = Event()
    modify_value = {}  # A dictionary of locks, one for each data location.
    location_locks = []

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device, spawning 8 worker threads.
        
        Args:
            device_id: A unique identifier for this device.
            sensor_data: A dictionary of the device's local sensor data.
            supervisor: An external object that provides neighborhood information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbours = None
        self.scripts = []
        self.timepoint_done = Event()
        self.data_lock = Lock()
        self.neighbours_lock = Lock()
        self.crt_scripts = LifoQueue() # Work is pushed onto a LIFO queue.
        self.devices = []
        self.ready = Event()
        self.my_data_lock = Lock()
        self.queue_lock = Lock()
        self.threads = []
        self.signal_end = False
        self.index = 0
        
        # All devices spawn a fixed number of worker threads.
        for i in range(8):
            self.threads.append(DeviceThread(self))
            self.threads[i].start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the global barrier for all devices in the simulation.
        This is expected to be called once on a single device (e.g., device 0).
        """
        self.devices = devices
        if self.device_id == 0:
            Device.barry = ReusableBarrierCond(len(devices) * 8)
            Device.barry_is_set.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be processed. A `None` script signals the end of a
        timepoint and triggers worker threads to proceed to the next barrier.
        """
        # Create a per-location lock if one doesn't exist.
        if location not in Device.modify_value:
            Device.modify_value[location] = Lock()

        if script is None:
            self.timepoint_done.set()
            # Push sentinels for all worker threads.
            for i in range(8):
                self.crt_scripts.put((None, None))
        else:
            self.crt_scripts.put((script, location))

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to complete."""
        for i in range(8):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread that executes scripts for a parent Device.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop, synchronized by a global barrier.
        """
        # Wait for the global barrier to be initialized.
        Device.barry_is_set.wait()
        while True:
            # --- GLOBAL SYNC POINT 1 ---
            # All threads from all devices wait here before starting a timepoint.
            Device.barry.wait()
            
            # The first thread for this device fetches the neighbor list.
            with self.device.neighbours_lock:
                if self.device.signal_end:
                    break # Break the loop if shutdown is signaled.
                if self.device.neighbours is None:
                    self.device.neighbours = self.device.supervisor.get_neighbours()
                    if self.device.neighbours is None:
                        self.device.signal_end = True
                        break

            # Block Logic: Process scripts from the work queue.
            # This is an unusual pattern where threads iterate over the queue using a
            # shared index instead of each thread popping an item.
            while True:
                with self.device.queue_lock:
                    if self.device.index < self.device.crt_scripts.qsize():
                        script, location = self.device.crt_scripts.queue[self.device.index]
                        if script is None:
                            # Sentinel value found, signaling end of work for this timepoint.
                            self.device.crt_scripts.get()
                            break
                        
                        # Acquire a global lock for the specific data location.
                        Device.modify_value[location].acquire()
                        self.device.index += 1
                
                # --- Critical Section for Script Execution ---
                script_data = []
                # Gather data from self and neighbors.
                with self.device.my_data_lock:
                    data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                for device in self.device.neighbours:
                    with device.my_data_lock:
                        data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                if script_data:
                    # Run script and broadcast result.
                    result = script.run(script_data)
                    if result:
                        for dev in self.device.neighbours:
                            with dev.my_data_lock:
                                dev.set_data(location, result)
                        with self.device.my_data_lock:
                            self.device.set_data(location, result)
                Device.modify_value[location].release()
            
            # --- GLOBAL SYNC POINT 2 ---
            # All threads wait here after processing their portion of the queue.
            Device.barry.wait()

            # The first thread resets the state for the next timepoint.
            with self.device.neighbours_lock:
                if self.device.neighbours is not None:
                    self.device.neighbours = None
                if self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
            
            with self.device.queue_lock:
                if self.device.index > 0:
                    self.device.index = 0
            
            # --- GLOBAL SYNC POINT 3 ---
            # Wait again to ensure all threads have completed cleanup before the next loop.
            Device.barry.wait()
