#- Architectural Intent: This file appears to be a concatenation of two separate modules.
#  1. A reimplementation of Python's standard thread-safe `Queue` module.
#  2. A custom `device` module that uses the queue for a device simulation.
#  The comments will address both sections.

"""
This module provides thread-safe queue implementations for producer-consumer problems.
It includes a standard FIFO Queue, a PriorityQueue, and a LifoQueue (stack).
The implementation is very similar to Python's standard library `Queue` module and
is built upon `threading.Lock` and `threading.Condition` for synchronization.
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
    """Exception raised by non-blocking get() and get_nowait() on an empty Queue."""
    pass

class Full(Exception):
    """Exception raised by non-blocking put() and put_nowait() on a full Queue."""
    pass

class Queue:
    """A thread-safe, FIFO (First-In, First-Out) queue.

    Producers can add items to the queue and consumers can retrieve them. If the
    queue is full, producers will block until space is available. If the queue
    is empty, consumers will block until an item is available.

    Attributes:
        maxsize (int): The maximum size of the queue. If 0, the queue is unbounded.
        mutex (threading.Lock): A lock for serializing access to the queue.
        not_empty (threading.Condition): A condition variable to signal when the queue is not empty.
        not_full (threading.Condition): A condition variable to signal when the queue is not full.
        all_tasks_done (threading.Condition): A condition to signal when all tasks are completed.
        unfinished_tasks (int): The number of tasks that are currently being processed.
    """
    def __init__(self, maxsize=0):
        """Initializes a new Queue instance."""
        self.maxsize = maxsize
        self._init(maxsize)
        
        #- Functional Utility: A mutex to protect access to the internal queue data structure.
        self.mutex = _threading.Lock()
        
        # Condition for producers to wait on when the queue is full.
        self.not_empty = _threading.Condition(self.mutex)
        
        # Condition for consumers to wait on when the queue is empty.
        self.not_full = _threading.Condition(self.mutex)
        
        # Condition for joining threads waiting for all tasks to be completed.
        self.all_tasks_done = _threading.Condition(self.mutex)
        self.unfinished_tasks = 0

    def task_done(self):
        """Indicate that a formerly enqueued task is complete.

        Used by consumer threads. For each get() that is used to fetch a task,
        a subsequent call to task_done() tells the queue that the processing
        on that task is complete.

        If a join() is currently blocking, it will resume when all items have been
        processed (meaning that a task_done() call was received for every item
        that had been put() into the queue).
        """
        self.all_tasks_done.acquire()
        try:
            unfinished = self.unfinished_tasks - 1
            if unfinished <= 0:
                if unfinished < 0:
                    raise ValueError('task_done() called too many times')
                # If all tasks are done, wake up any threads waiting in join().
                self.all_tasks_done.notify_all()
            self.unfinished_tasks = unfinished
        finally:
            self.all_tasks_done.release()

    def join(self):
        """Blocks until all items in the queue have been gotten and processed.

        The count of unfinished tasks goes up whenever an item is added to the
        queue. The count goes down whenever a consumer thread calls task_done().
        When the count of unfinished tasks drops to zero, join() unblocks.
        """
        self.all_tasks_done.acquire()
        try:
            # Block as long as there are unfinished tasks.
            while self.unfinished_tasks:
                self.all_tasks_done.wait()
        finally:
            self.all_tasks_done.release()

    def qsize(self):
        """Return the approximate size of the queue (not reliable)."""
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

    def empty(self):
        """Return True if the queue is empty, False otherwise (not reliable)."""
        self.mutex.acquire()
        n = not self._qsize()
        self.mutex.release()
        return n

    def full(self):
        """Return True if the queue is full, False otherwise (not reliable)."""
        self.mutex.acquire()
        n = 0 < self.maxsize == self._qsize()
        self.mutex.release()
        return n

    def put(self, item, block=True, timeout=None):
        """Put an item into the queue.

        Args:
            item: The item to be put into the queue.
            block (bool): If True (the default), block until a free slot is
                available. If False, raise Full exception if the queue is full.
            timeout (float): The timeout in seconds for blocking. If None,
                block indefinitely.
        Raises:
            Full: If the queue is full and blocking is False.
        """
        self.not_full.acquire()
        try:
            # Pre-condition: Check if the queue is full and handle blocking/timeout logic.
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
            
            # Block Logic: Add the item to the queue and increment task counter.
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify() # Signal that the queue is no longer empty.
        finally:
            self.not_full.release()

    def put_nowait(self, item):
        """Put an item into the queue without blocking.

        Equivalent to put(item, False).
        """
        return self.put(item, False)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue.

        Args:
            block (bool): If True (the default), block until an item is
                available. If False, raise Empty exception if the queue is empty.
            timeout (float): The timeout in seconds for blocking. If None,
                block indefinitely.
        Raises:
            Empty: If the queue is empty and blocking is False.
        """
        self.not_empty.acquire()
        try:
            # Pre-condition: Check if the queue is empty and handle blocking/timeout logic.
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
            
            # Block Logic: Retrieve an item from the queue.
            item = self._get()
            self.not_full.notify() # Signal that the queue is no longer full.
            return item
        finally:
            self.not_empty.release()

    def get_nowait(self):
        """Remove and return an item from the queue without blocking.

        Equivalent to get(False).
        """
        return self.get(False)

    
    #--- Internal methods for queue implementation ---

    
    def _init(self, maxsize):
        """Initialize the internal queue representation."""
        self.queue = deque()

    def _qsize(self, len=len):
        """Return the size of the internal queue."""
        return len(self.queue)

    
    def _put(self, item):
        """Put an item into the internal queue."""
        self.queue.append(item)

    
    def _get(self):
        """Get an item from the internal queue."""
        return self.queue.popleft()


class PriorityQueue(Queue):
    """A thread-safe priority queue.

    Items are inserted in any order. `get()` returns the item with the lowest
    value first (min-heap). Items are typically tuples like (priority, data).
    """

    def _init(self, maxsize):
        """Initialize the internal queue as a list for the heap."""
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item, heappush=heapq.heappush):
        """Add an item to the heap."""
        heappush(self.queue, item)

    def _get(self, heappop=heapq.heappop):
        """Remove and return the smallest item from the heap."""
        return heappop(self.queue)


class LifoQueue(Queue):
    """A thread-safe LIFO (Last-In, First-Out) queue, also known as a stack."""

    def _init(self, maxsize):
        """Initialize the internal queue as a list."""
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        """Put an item on top of the stack."""
        self.queue.append(item)

    def _get(self):
        """Remove and return the most recently added item from the stack."""
        return self.queue.pop()

# --- End of Queue module implementation ---
# --- Start of device module implementation ---

# Note: The `barrier` module is not provided in this file.
# from barrier import ReusableBarrierSem 

class Device(object):
    """Represents a device node in a simulated distributed environment.

    Each device runs its own threads, processes scripts from a workpool,
    and can communicate with neighboring devices to get/set data.
    Synchronization between devices is managed via a shared barrier.
    """
    
    #- Class Attribute: A shared barrier for synchronizing all device instances.
    devices_barrier = None
    nr_devices = 0

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""

        Device.nr_devices += 1
        self.current_neighbours = None
        self.current_neighbours_set = False
        self.access_neighbours = Lock()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.workpool = Queue() # A queue for scripts to be executed.
        self.timepoint_done = Event()
        self.threads = []
        self.passed = True
        self.passed_lock = Lock()
        self.device_barrier = None
        self.threads_barrier = ReusableBarrierSem(1) # Barrier for internal threads.
        self.start_barrier = ReusableBarrierSem(1)
        
        #- Block Logic: Start the initial worker thread for this device.
        thread = DeviceThread(self, 1)
        self.threads.append(thread)
        thread.start()

    @classmethod
    def get_devices_barrier(cls):
        """Lazily initializes and returns the shared barrier for all devices."""
        if cls.devices_barrier == None:
            cls.devices_barrier = ReusableBarrierSem(cls.nr_devices)
        return cls.devices_barrier

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the shared barrier among all devices."""
        self.device_barrier = Device.get_devices_barrier()

    def assign_script(self, script, location):
        """Assigns a new script to be executed by this device.

        If the number of scripts exceeds the number of threads, a new worker
        thread is spawned to handle the increased load.
        """
        if script is not None:
            self.workpool.put((script, location))
            self.scripts.append((script, location))

            #- Block Logic: Dynamically scale the number of worker threads.
            if len(self.scripts) > len(self.threads) & len(self.threads) < 9:
                self.threads_barrier = ReusableBarrierSem(len(self.threads) + 1)
                thread = DeviceThread(self, len(self.threads) + 1)
                self.threads.append(thread)
                thread.start()
            self.script_received.set()

        else:
            #- Inline: A None script is a signal to end the current timepoint.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific location in the device's sensor data map."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its worker threads."""
        for thread in self.threads:
            thread.join()

class DeviceThread(Thread):
    """A worker thread for a Device instance.

    This thread waits for scripts to be assigned, executes them using data from
    neighboring devices, and synchronizes with other threads and devices at the
    end of each time step.
    """
    def __init__(self, device, name):
        Thread.__init__(self)
        self.device = device
        self.name = "%d" % device.device_id + "%d" % name

    def run(self):
        """The main loop for the worker thread."""
        while True:
            # Block Logic: The first thread to pass this lock fetches the new neighbor list for the current time step.
            with self.device.passed_lock:
                if self.device.passed == True:
                    self.device.passed = False
                    self.device.current_neighbours = self.device.supervisor.get_neighbours()

            # A None neighbor list is a signal to shut down.
            if self.device.current_neighbours is None:
                break
            else:
                self.device.script_received.wait() # Wait for supervisor to assign scripts.
                while True:
                    try:
                        # Process all scripts in the workpool for this time step.
                        (script, location) = self.device.workpool.get_nowait()
                    except Exception:
                        break # Break if workpool is empty.
                    
                    script_data = []
                    
                    # Block Logic: Gather data from all neighbors for the script execution.
                    try:
                        for device in self.device.current_neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                    except Exception:
                        break # Exit if a neighbor-related error occurs.
                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                    
                    # Block Logic: Run the script and broadcast the result to self and neighbors.
                    if script_data != []:
                        result = script.run(script_data)
                        try:
                            for device in self.device.current_neighbours:
                                device.set_data(location, result)
                        except Exception:
                            break
                        self.device.set_data(location, result)
            
            # --- Synchronization Point ---
            self.device.timepoint_done.wait() # Wait until the supervisor signals the end of the timepoint.
            self.device.threads_barrier.wait() # All internal threads sync up.
            self.device.timepoint_done.clear()

            # Block Logic: The first thread to pass here waits for all other devices at a global barrier,
            # then resets the state for the next time step.
            with self.device.passed_lock:
                if self.device.passed != True:
                    self.device.device_barrier.wait()
                    self.device.passed = True
                    
                    # Re-queue persistent scripts for the next time step.
                    for (script,location) in self.device.scripts:
                        self.device.workpool.put((script,location))
        
        #--- Shutdown Logic ---
        self.device.threads_barrier.stop_barrier()
        self.device.devices_barrier.stop_barrier()
        self.device.timepoint_done.set()
