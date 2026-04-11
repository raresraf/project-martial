"""
This file contains implementations for two distinct components:
1. A thread-safe queue library, similar to Python's standard `Queue` module.
   It provides FIFO, LIFO, and Priority Queue implementations.
2. A device simulation framework, which appears to model a network of
   interconnected devices that process sensor data based on distributed scripts.

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
    multiple threads to safely add (`put`) and remove (`get`) items. It supports
    blocking operations, timeouts, and tracking of unfinished tasks, making it
    a fundamental tool for producer-consumer problems in concurrent programming.

    Attributes:
        maxsize (int): The maximum size of the queue. If 0, the queue is infinite.
        mutex (Lock): A lock to protect the internal state of the queue.
        not_empty (Condition): A condition variable signaled when an item is added
                               to an empty queue.
        not_full (Condition): A condition variable signaled when an item is removed
                              from a full queue.
        all_tasks_done (Condition): A condition variable signaled when all tasks
                                    are completed.
        unfinished_tasks (int): The number of tasks yet to be completed.
    """
    def __init__(self, maxsize=0):
        """Initializes a new Queue instance."""
        self.maxsize = maxsize
        self._init(maxsize)
        
        self.mutex = _threading.Lock()
        
        # Condition for waiting until the queue is not empty.
        self.not_empty = _threading.Condition(self.mutex)
        
        # Condition for waiting until the queue is not full.
        self.not_full = _threading.Condition(self.mutex)
        
        # Condition for waiting until all tasks are done.
        self.all_tasks_done = _threading.Condition(self.mutex)
        self.unfinished_tasks = 0

    def task_done(self):
        """
        Indicate that a formerly enqueued task is complete.

        Used by consumer threads. For each `get()` used to fetch a task,
        a subsequent call to `task_done()` tells the queue that the processing
        on the task is complete. If `join()` is currently blocking, it will resume
        when all items have been processed.
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

        The count of unfinished tasks goes up whenever an item is added to the
        queue. The count goes down whenever a consumer thread calls `task_done()`.
        When the count of unfinished tasks drops to zero, `join()` unblocks.
        """
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
        """
        Put an item into the queue.

        If `block` is True and `timeout` is None (the default), block until a
        free slot is available. If `timeout` is a positive number, it blocks
        at most `timeout` seconds and raises the `Full` exception if no free
        slot was available within that time. Otherwise (`block` is False), put an
        item on the queue if a free slot is immediately available, else raise
        the `Full` exception.
        """
        self.not_full.acquire()
        try:
            if self.maxsize > 0:
                if not block:
                    if self._qsize() == self.maxsize:
                        raise Full
                elif timeout is None:
                    # Block indefinitely until a slot is free.
                    while self._qsize() == self.maxsize:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    # Block with a timeout.
                    endtime = _time() + timeout
                    while self._qsize() == self.maxsize:
                        remaining = endtime - _time()
                        if remaining <= 0.0:
                            raise Full
                        self.not_full.wait(remaining)
            self._put(item)
            self.unfinished_tasks += 1
            # Signal that the queue is no longer empty.
            self.not_empty.notify()
        finally:
            self.not_full.release()

    def put_nowait(self, item):
        """Equivalent to put(item, False)."""
        return self.put(item, False)

    def get(self, block=True, timeout=None):
        """
        Remove and return an item from the queue.

        If `block` is True and `timeout` is None (the default), block until an
        item is available. If `timeout` is a positive number, it blocks at
        most `timeout` seconds and raises the `Empty` exception if no item was
        available within that time. Otherwise (`block` is False), return an
        item if one is immediately available, else raise the `Empty` exception.
        """
        self.not_empty.acquire()
        try:
            if not block:
                if not self._qsize():
                    raise Empty
            elif timeout is None:
                # Block indefinitely until an item is available.
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                # Block with a timeout.
                endtime = _time() + timeout
                while not self._qsize():
                    remaining = endtime - _time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            item = self._get()
            # Signal that the queue is no longer full.
            self.not_full.notify()
            return item
        finally:
            self.not_empty.release()

    def get_nowait(self):
        """Equivalent to get(False)."""
        return self.get(False)

    # --- Internal methods for concrete queue implementations ---
    def _init(self, maxsize):
        """Initialize the internal queue data structure (a deque for FIFO)."""
        self.queue = deque()

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        """Add an item to the right side of the deque."""
        self.queue.append(item)

    def _get(self):
        """Remove and return an item from the left side of the deque."""
        return self.queue.popleft()


class PriorityQueue(Queue):
    """
    A thread-safe priority queue.

    Items are inserted in any order. `get()` will retrieve the item with the
    lowest value first (sorted by the item itself). It uses a heap (min-heap)
    to efficiently manage the order.
    """
    def _init(self, maxsize):
        """Initialize the queue as a list, which will be managed as a heap."""
        self.queue = []

    def _put(self, item, heappush=heapq.heappush):
        """Add an item to the heap."""
        heappush(self.queue, item)

    def _get(self, heappop=heapq.heappop):
        """Remove and return the smallest item from the heap."""
        return heappop(self.queue)


class LifoQueue(Queue):
    """
    A thread-safe LIFO (Last-In, First-Out) queue, also known as a Stack.
    """
    def _init(self, maxsize):
        """Initialize the queue as a standard list."""
        self.queue = []

    def _put(self, item):
        """Append an item to the end of the list."""
        self.queue.append(item)

    def _get(self):
        """Remove and return the last item from the list."""
        return self.queue.pop()


# =============================================================================
# Part 2: Device Simulation Framework
# =============================================================================

from threading import Event, Thread, Lock, Condition
# Assuming `barrier` is a custom module providing a ReusableBarrierSem.
# from barrier import ReusableBarrierSem

class ReusableBarrierSem:
    """A placeholder for a custom ReusableBarrierSemaphore."""
    def __init__(self, num_threads):
        pass
    def wait(self):
        pass
    def stop_barrier(self):
        pass


class Device(object):
    """
    Represents a single device (or node) in a simulated sensor network.

    Each device has its own sensor data, a list of neighbors, and a workpool
    of scripts to execute. It manages multiple worker threads (`DeviceThread`)
    to process these scripts. The class uses a complex system of locks, events,
    and barriers to coordinate actions with other devices and its own threads
    across discrete timepoints.
    """
    devices_barrier = None
    nr_devices = 0

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id: A unique identifier for this device.
            sensor_data: A dictionary representing the device's local sensor readings.
            supervisor: An object responsible for providing neighborhood information.
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
        self.workpool = Queue() # Workpool for (script, location) tuples.
        self.timepoint_done = Event()
        self.threads = []
        self.passed = True
        self.passed_lock = Lock()
        self.device_barrier = None
        # Barrier for coordinating this device's own worker threads.
        self.threads_barrier = ReusableBarrierSem(1)
        self.start_barrier = ReusableBarrierSem(1)
        # Start the initial worker thread.
        thread = DeviceThread(self, 1)
        self.threads.append(thread)
        thread.start()

    @classmethod
    def get_devices_barrier(cls):
        """A class method to get a shared barrier for all device instances."""
        if cls.devices_barrier == None:
            cls.devices_barrier = ReusableBarrierSem(cls.nr_devices)
        return cls.devices_barrier

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes the shared barrier for all devices."""
        self.device_barrier = Device.get_devices_barrier()

    def assign_script(self, script, location):
        """
        Assigns a new script to be processed by this device.

        If a script is provided, it's added to the workpool. If the number of
        scripts exceeds the number of threads, a new worker thread is spawned.
        If the script is None, it signals the end of a timepoint.
        """
        if script is not None:
            self.workpool.put((script, location))
            self.scripts.append((script, location))

            # Spawn a new worker thread if needed.
            if len(self.scripts) > len(self.threads) and len(self.threads) < 9:
                self.threads_barrier = ReusableBarrierSem(len(self.threads) + 1)
                thread = DeviceThread(self, len(self.threads) + 1)
                self.threads.append(thread)
                thread.start()
            self.script_received.set()
        else:
            # A None script signals the end of the current timepoint's script assignments.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location from this device."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location on this device."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to complete."""
        for thread in self.threads:
            thread.join()

class DeviceThread(Thread):
    """
    A worker thread for a Device.

    This thread's main responsibility is to execute scripts from the device's
    workpool. It coordinates with other threads and devices using a system of
    barriers and events to ensure that operations happen in synchronized
    timepoints.
    """
    def __init__(self, device, name):
        Thread.__init__(self)
        self.device = device
        self.name = "%d" % device.device_id + "%d" % name

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            # Block Logic: Main synchronization point. The first thread to pass
            # this lock fetches the new list of neighbors for the current timepoint.
            with self.device.passed_lock:
                if self.device.passed == True:
                    self.device.passed = False
                    self.device.current_neighbours = self.device.supervisor.get_neighbours()

            if self.device.current_neighbours is None:
                # A None neighbor list is the signal to terminate.
                break
            else:
                # Wait until all scripts for the current timepoint have been assigned.
                self.device.script_received.wait()
                # Invariant: Process all scripts currently in the workpool.
                while True:
                    try:
                        (script, location) = self.device.workpool.get_nowait()
                    except Empty:
                        # Workpool is empty, break to wait for the next timepoint.
                        break
                    
                    script_data = []
                    # Block Logic: Gather data from all neighbors for the script's location.
                    try:
                        for device in self.device.current_neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                    except Exception:
                        break # Likely a shutdown signal.
                    
                    # Also gather data from the local device.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                    
                    if script_data:
                        # Execute the script with the collected data.
                        result = script.run(script_data)
                        
                        # Block Logic: Broadcast the result to all neighbors.
                        try:
                            for device in self.device.current_neighbours:
                                device.set_data(location, result)
                        except Exception:
                            break # Likely a shutdown signal.
                        
                        self.device.set_data(location, result)
            
            # Synchronization: Wait for the signal that the timepoint is done.
            self.device.timepoint_done.wait()
            # Synchronization: All worker threads for this device wait here.
            self.device.threads_barrier.wait()
            self.device.timepoint_done.clear()

            # Synchronization: The first thread to pass this lock will wait at the
            # global device barrier, ensuring all devices are in sync.
            with self.device.passed_lock:
                if self.device.passed != True:
                    # Global barrier for all devices in the simulation.
                    self.device.device_barrier.wait()
                    self.device.passed = True
                    
                    # Re-queue all scripts for the next timepoint.
                    for (script,location) in self.device.scripts:
                        self.device.workpool.put((script,location))
        
        # Stop all barriers to allow any waiting threads to terminate.
        self.device.threads_barrier.stop_barrier()
        self.device.devices_barrier.stop_barrier()
        self.device.timepoint_done.set()
