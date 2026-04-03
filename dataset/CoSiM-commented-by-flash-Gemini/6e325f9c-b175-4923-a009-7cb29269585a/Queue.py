




"""
@file Queue.py
@brief Provides thread-safe queue implementations: `Queue`, `PriorityQueue`, and `LifoQueue`.

This module also includes `Device` and `DeviceThread` classes, which are likely part of a
distributed simulation or task processing framework. The `Device` class manages device-specific
data, scripts, and communication, while `DeviceThread` executes scripts on behalf of a device.

Domain: Concurrency, Data Structures, Distributed Systems (Simulation).
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
    @brief Exception raised by Queue.get(block=0)/get_nowait() when the queue is empty.
    """
    pass

class Full(Exception):
    """
    @brief Exception raised by Queue.put(block=0)/put_nowait() when the queue is full.
    """
    pass

class Queue:
    """
    @brief A thread-safe queue implementation, inspired by Python's `queue` module.
    
    This class provides a synchronized queue that can be used to safely exchange
    data between multiple threads. It supports optional size limits, blocking/non-blocking
    operations, and task tracking (`task_done`, `join`).
    """
    
    def __init__(self, maxsize=0):
        """
        @brief Initializes a new thread-safe Queue instance.
        @param maxsize The maximum number of items allowed in the queue. If 0 (default), the queue size is infinite.
        
        Initializes the queue's internal state, including:
        - `maxsize`: The maximum capacity of the queue.
        - `_init(maxsize)`: Calls an internal method to set up the underlying data structure (deque).
        - `mutex`: A reentrant lock to protect the queue's internal state.
        - `not_empty`: A condition variable for consumers to wait when the queue is empty.
        - `not_full`: A condition variable for producers to wait when the queue is full.
        - `all_tasks_done`: A condition variable for tracking completion of submitted tasks.
        - `unfinished_tasks`: A counter for tasks put into the queue but not yet marked done.
        """
        self.maxsize = maxsize
        self._init(maxsize)
        
        
        
        
        self.mutex = _threading.Lock()
        
        
        self.not_empty = _threading.Condition(self.mutex)
        
        
        self.not_full = _threading.Condition(self.mutex)
        
        
        self.all_tasks_done = _threading.Condition(self.mutex)
        self.unfinished_tasks = 0

    def task_done(self):
        """
        @brief Indicates that a formerly enqueued task is complete.
        
        This method is used by queue consumers to signal that a task retrieved
        from the queue has been fully processed. It decrements the `unfinished_tasks`
        counter. If `unfinished_tasks` drops to zero, all threads waiting on `join()`
        are notified.
        
        @raises ValueError: If called more times than there were items placed in the queue.
        """
        self.all_tasks_done.acquire()
        try:
            unfinished = self.unfinished_tasks - 1
            if unfinished <= 0:
                if unfinished < 0:
                    raise ValueError('task_done() called too many times')
                # Block Logic: If all tasks are done, notify waiting threads.
                self.all_tasks_done.notify_all()
            self.unfinished_tasks = unfinished
        finally:
            self.all_tasks_done.release()

    def join(self):
        """
        @brief Blocks until all items in the queue have been gotten and processed.
        
        The `join()` method blocks the calling thread until `task_done()` has been
        called for every item previously put into the queue.
        """
        self.all_tasks_done.acquire()
        try:
            # Block Logic: Wait while there are still unfinished tasks.
            while self.unfinished_tasks:
                self.all_tasks_done.wait()
        finally:
            self.all_tasks_done.release()

    def qsize(self):
        """
        @brief Returns the approximate number of items in the queue.
        @return The approximate size of the queue.
        """
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

    def empty(self):
        """
        @brief Returns True if the queue is empty, False otherwise.
        @return True if the queue is empty, False otherwise.
        """
        self.mutex.acquire()
        n = not self._qsize()
        self.mutex.release()
        return n

    def full(self):
        """
        @brief Returns True if the queue is full, False otherwise.
        @return True if the queue is full, False otherwise.
        
        A Queue is considered full if its `maxsize` is greater than 0
        and the number of items in the queue equals `maxsize`.
        """
        self.mutex.acquire()
        n = 0 < self.maxsize == self._qsize()
        self.mutex.release()
        return n

    def put(self, item, block=True, timeout=None):
        """
        @brief Puts an item into the queue.
        @param item The item to be added to the queue.
        @param block If True (default), block until a free slot is available.
        @param timeout If block is True, wait at most `timeout` seconds.
        @raises Full: If block is False and the queue is full.
        @raises ValueError: If timeout is negative.
        
        If the queue is full and `block` is True, this method will block
        until a slot becomes available. If `timeout` is specified, it will
        wait for at most that many seconds.
        """
        self.not_full.acquire()
        try:
            # Block Logic: Handle queues with a maximum size.
            if self.maxsize > 0:
                # Block Logic: Non-blocking put operation.
                if not block:
                    if self._qsize() == self.maxsize:
                        raise Full
                # Block Logic: Blocking put operation with no timeout.
                elif timeout is None:
                    while self._qsize() == self.maxsize:
                        self.not_full.wait()
                # Block Logic: Blocking put operation with a specified timeout.
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = _time() + timeout
                    while self._qsize() == self.maxsize:
                        remaining = endtime - _time()
                        if remaining <= 0.0:
                            raise Full
                        self.not_full.wait(remaining)
            # Block Logic: Atomically add the item to the internal queue and increment task counter.
            self._put(item)
            self.unfinished_tasks += 1
            # Block Logic: Notify any waiting consumers that the queue is no longer empty.
            self.not_empty.notify()
        finally:
            self.not_full.release()

    def put_nowait(self, item):
        """
        @brief Puts an item into the queue without blocking.
        @param item The item to be added to the queue.
        @raises Full: If the queue is full.
        @return The result of `put(item, False)`.
        """
        return self.put(item, False)

    def get(self, block=True, timeout=None):
        """
        @brief Retrieves an item from the queue.
        @param block If True (default), block until an item is available.
        @param timeout If block is True, wait at most `timeout` seconds.
        @raises Empty: If block is False and the queue is empty.
        @raises ValueError: If timeout is negative.
        @return The item retrieved from the queue.
        
        If the queue is empty and `block` is True, this method will block
        until an item becomes available. If `timeout` is specified, it will
        wait for at most that many seconds.
        """
        self.not_empty.acquire()
        try:
            # Block Logic: Non-blocking get operation.
            if not block:
                if not self._qsize():
                    raise Empty
            # Block Logic: Blocking get operation with no timeout.
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            # Block Logic: Blocking get operation with a specified timeout.
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = _time() + timeout
                while not self._qsize():
                    remaining = endtime - _time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            # Block Logic: Atomically retrieve the item from the internal queue.
            item = self._get()
            # Block Logic: Notify any waiting producers that the queue is no longer full.
            self.not_full.notify()
            return item
        finally:
            self.not_empty.release()

    def get_nowait(self):
        """
        @brief Retrieves an item from the queue without blocking.
        @raises Empty: If the queue is empty.
        @return The result of `get(False)`.
        """
        return self.get(False)

    
    
    
    # These methods are to be overridden by subclasses
    # in order to implement the specific queue semantics.

    def _init(self, maxsize):
        """
        @brief Internal method to initialize the queue's underlying data structure.
        @param maxsize The maximum size of the queue.
        
        This default implementation uses a `collections.deque`.
        """
        self.queue = deque()

    def _qsize(self, len=len):
        """
        @brief Internal method to return the size of the queue.
        @param len Built-in len function for efficiency.
        @return The number of items in the queue.
        """
        return len(self.queue)

    
    def _put(self, item):
        """
        @brief Internal method to put an item into the queue.
        @param item The item to be added.
        
        This default implementation appends the item to the `deque`.
        """
        self.queue.append(item)

    
    def _get(self):
        """
        @brief Internal method to get an item from the queue.
        @return The item removed from the queue.
        
        This default implementation removes and returns the leftmost item from the `deque`.
        """
        return self.queue.popleft()


class PriorityQueue(Queue):
    """
    @brief A thread-safe priority queue implementation.
    
    This class extends `Queue` to provide priority queue functionality.
    Items are retrieved based on their priority (smallest first).
    It uses the `heapq` module for its underlying data structure.
    """
    

    def _init(self, maxsize):
        """
        @brief Internal method to initialize the priority queue's underlying data structure.
        @param maxsize The maximum size of the queue (though not directly enforced by heapq).
        
        Initializes an empty list that will be used as a min-heap by the `heapq` module.
        """
        self.queue = []

    def _qsize(self, len=len):
        """
        @brief Internal method to return the size of the priority queue.
        @param len Built-in len function for efficiency.
        @return The number of items in the priority queue.
        """
        return len(self.queue)

    def _put(self, item, heappush=heapq.heappush):
        """
        @brief Internal method to put an item into the priority queue.
        @param item The item to be added.
        @param heappush The `heapq.heappush` function for efficiency.
        
        Adds an item to the heap, maintaining the heap invariant.
        """
        heappush(self.queue, item)

    def _get(self, heappop=heapq.heappop):
        """
        @brief Internal method to get an item from the priority queue.
        @param heappop The `heapq.heappop` function for efficiency.
        @return The smallest item from the priority queue.
        
        Removes and returns the smallest item from the heap.
        """
        return heappop(self.queue)


class LifoQueue(Queue):
    """
    @brief A thread-safe Last-In/First-Out (LIFO) queue implementation.
    
    This class extends `Queue` to provide LIFO queue functionality,
    behaving like a stack. The most recently added item is retrieved first.
    """
    

    def _init(self, maxsize):
        """
        @brief Internal method to initialize the LIFO queue's underlying data structure.
        @param maxsize The maximum size of the queue.
        
        Initializes an empty list that will be used for LIFO operations.
        """
        self.queue = []

    def _qsize(self, len=len):
        """
        @brief Internal method to return the size of the LIFO queue.
        @param len Built-in len function for efficiency.
        @return The number of items in the LIFO queue.
        """
        return len(self.queue)

    def _put(self, item):
        """
        @brief Internal method to put an item into the LIFO queue.
        @param item The item to be added.
        
        Appends an item to the end of the list, making it the last in.
        """
        self.queue.append(item)

    def _get(self):
        """
        @brief Internal method to get an item from the LIFO queue.
        @return The item removed from the queue.
        
        Removes and returns the last item appended to the list (Last-In, First-Out).
        """
class LifoQueue(Queue):
    """
    @brief A thread-safe Last-In/First-Out (LIFO) queue implementation.
    
    This class extends `Queue` to provide LIFO queue functionality,
    behaving like a stack. The most recently added item is retrieved first.
    """
    

    def _init(self, maxsize):
        """
        @brief Internal method to initialize the LIFO queue's underlying data structure.
        @param maxsize The maximum size of the queue.
        
        Initializes an empty list that will be used for LIFO operations.
        """
        self.queue = []

    def _qsize(self, len=len):
        """
        @brief Internal method to return the size of the LIFO queue.
        @param len Built-in len function for efficiency.
        @return The number of items in the LIFO queue.
        """
        return len(self.queue)

    def _put(self, item):
        """
        @brief Internal method to put an item into the LIFO queue.
        @param item The item to be added.
        
        Appends an item to the end of the list, making it the last in.
        """
        self.queue.append(item)

    def _get(self):
        """
        @brief Internal method to get an item from the LIFO queue.
        @return The item removed from the queue.
        
        Removes and returns the last item appended to the list (Last-In, First-Out).
        """
        return self.queue.pop()
# Start of concatenated file "device.py"
"""
@file device.py
@brief Implements a simulated distributed device and its associated thread for executing scripts.

This module is designed to simulate a device in a distributed system, handling sensor data,
communicating with a supervisor, and executing scripts based on received work.
It utilizes thread-safe queues and barriers for synchronization.
"""

from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierSem # @brief Imports a reusable barrier synchronization primitive.
import time
from Queue import * # @brief Imports the custom Queue implementations (Queue, PriorityQueue, LifoQueue).

class Device(object):
    """
    @brief Represents a simulated device in a distributed system.
    
    Each device has a unique ID, holds sensor data, interacts with a supervisor,
    and can execute scripts in a multi-threaded manner. It uses barriers for
    synchronization and queues for managing work.
    """
    devices_barrier = None # @brief Class-level barrier to synchronize all Device instances. Initialized lazily.
    nr_devices = 0 # @brief Class-level counter for the total number of Device instances created.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id The unique identifier for this device.
        @param sensor_data A dictionary or similar structure holding the device's sensor data.
        @param supervisor An object representing the supervisor controlling this device.
        
        Initializes device-specific attributes, locks, events, and starts a worker thread.
        """
        # Block Logic: Increment the class-level counter for total number of devices.
        Device.nr_devices += 1
        self.current_neighbours = None # @brief Stores the current list of neighboring devices.
        self.current_neighbours_set = False # @brief Flag indicating if neighbors have been set for the current timepoint.
        self.access_neighbours = Lock() # @brief Lock to protect access to `current_neighbours`.
        self.device_id = device_id # @brief Unique identifier for this device.
        self.sensor_data = sensor_data # @brief Data collected by this device's sensors.
        self.supervisor = supervisor # @brief Reference to the central supervisor managing devices.
        self.script_received = Event() # @brief Event to signal when a new script has been assigned.
        self.scripts = [] # @brief List of scripts assigned to this device.
        self.workpool = Queue() # @brief Thread-safe queue for scripts to be executed.
        self.timepoint_done = Event() # @brief Event to signal that a timepoint's work is done.
        self.threads = [] # @brief List of worker threads associated with this device.
        self.passed = True # @brief Flag indicating if the device has 'passed' a certain check or phase.
        self.passed_lock = Lock() # @brief Lock to protect the `passed` flag.
        self.device_barrier = None # @brief Barrier specific to this device for timepoint synchronization.
        self.threads_barrier = ReusableBarrierSem(1) # @brief Barrier to synchronize threads within this device.
        self.start_barrier = ReusableBarrierSem(1) # @brief Barrier for initial thread startup.
        # Block Logic: Create and start the initial worker thread for this device.
        thread = DeviceThread(self, 1)
        self.threads.append(thread)
        thread.start()

    @classmethod
    def get_devices_barrier(cls):
        """
        @brief Returns the class-level barrier for all devices, initializing it if necessary.
        @return A `ReusableBarrierSem` instance for synchronizing all `Device` instances.
        
        This barrier ensures that all devices start processing a new timepoint simultaneously.
        """
        # Block Logic: Lazily initialize the shared barrier once all devices are counted.
        if cls.devices_barrier == None:
            cls.devices_barrier = ReusableBarrierSem(cls.nr_devices)
        return cls.devices_barrier

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device %d" % self.device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up synchronization barriers for the device after all devices are initialized.
        @param devices A list or collection of all `Device` instances in the simulation.
        
        This method is intended to be called after all devices have been created to
        properly initialize the device-specific barrier which relies on the total
        number of devices.
        """
        self.device_barrier = Device.get_devices_barrier()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data location.
        @param script The script object to be executed.
        @param location The data location (e.g., sensor ID) where the script should operate.
        
        Adds the script to the device's work queue. If the number of scripts exceeds
        the current number of worker threads (and is within a limit), a new worker
        thread is spawned. Signals that a script has been received, unblocking
        worker threads that might be waiting for work.
        """
        if script is not None:
            # Block Logic: Add the script and its location to the workpool queue.
            self.workpool.put((script, location))
            # Block Logic: Keep a record of assigned scripts.
            self.scripts.append((script, location))

            # Block Logic: Dynamically scale worker threads based on the number of scripts.
            # Invariant: The number of threads does not exceed a hardcoded limit (9).
            if len(self.scripts) > len(self.threads) and len(self.threads) < 9:
                # Block Logic: Reinitialize the thread barrier to include the new thread.
                self.threads_barrier = ReusableBarrierSem(len(self.threads) + 1)
                thread = DeviceThread(self, len(self.threads) + 1)
                self.threads.append(thread)
                thread.start()
            # Block Logic: Signal that a script has been received, unblocking waiting threads.
            self.script_received.set()

        else:
            # Block Logic: If no script is assigned (signaling end of scripts for a timepoint),
            # signal completion events.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location The specific location (e.g., sensor ID) for which to retrieve data.
        @return The sensor data if `location` exists, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
        @param location The specific location (e.g., sensor ID) for which to set data.
        @param data The new data value to be set.
        
        Updates the device's sensor data for the specified location if it already exists.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down all worker threads associated with this device.
        
        Iterates through all `DeviceThread` instances and calls `join()` on each
        to ensure they complete their tasks before the device is fully shut down.
        """
        for thread in self.threads:
            thread.join()

class DeviceThread(Thread):
    """
    @brief Represents a worker thread that executes scripts for a `Device`.
    
    Each `DeviceThread` runs in a continuous loop, fetching scripts from its
    associated `Device`'s workpool, executing them, and handling synchronization
    with other threads and devices.
    """

    def __init__(self, device, name):
        """
        @brief Initializes a new `DeviceThread` instance.
        @param device The `Device` instance that this thread belongs to.
        @param name A unique name or identifier for this thread within the device.
        
        Initializes the thread and sets its associated device and name.
        """
        Thread.__init__(self)
        self.device = device
        self.name = "%d" % device.device_id + "%d" % name

    def run(self):
        """
        @brief The main execution loop for the `DeviceThread`.
        
        This method continuously processes scripts assigned to its device.
        It handles synchronization, retrieves neighboring device data, executes
        scripts, and updates device data. It also manages timepoint synchronization
        and barrier waits. The loop breaks if the device has no current neighbors (signaling shutdown).
        """
        while True:
            # Block Logic: Acquire lock to check and potentially update the 'passed' state and neighbors.
            with self.device.passed_lock:
                # Pre-condition: If the device has passed a check, reset it and fetch new neighbors.
                if self.device.passed == True:
                    self.device.passed = False
                    self.device.current_neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: If no current neighbors are available, it signifies the end of the simulation for this device.
            if self.device.current_neighbours is None:
                # Post-condition: Break the infinite loop, allowing the thread to terminate.
                break
            else:
                # Block Logic: Wait until a script is received by the device.
                self.device.script_received.wait()
                while True:
                    try:
                        # Block Logic: Attempt to get a script from the workpool without blocking.
                        (script, location) = self.device.workpool.get_nowait()
                    except Exception:
                        # Pre-condition: If no scripts are available, exit the inner loop.
                        # This also indicates that all scripts for the current timepoint have been processed.
                        break
                    script_data = []
                    
                    try:
                        # Block Logic: Collect data from neighboring devices for the script.
                        for device in self.device.current_neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                    except Exception:
                        # Pre-condition: Handle potential errors during neighbor data retrieval.
                        break
                    
                    # Block Logic: Collect data from the current device itself.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                    # Block Logic: If there is data to process, run the script.
                    if script_data != []:
                        # Functional Utility: Execute the assigned script with collected data.
                        result = script.run(script_data)
                        
                        try:
                            # Block Logic: Propagate the script's result to neighboring devices.
                            for device in self.device.current_neighbours:
                                device.set_data(location, result)
                        except Exception:
                            # Pre-condition: Handle potential errors during neighbor data update.
                            break
                        
                        # Block Logic: Update the current device's data with the script's result.
                        self.device.set_data(location, result)
            
            # Block Logic: Wait for the current timepoint's work to be explicitly marked as done.
            self.device.timepoint_done.wait()
            # Block Logic: Synchronize with other threads within the same device.
            self.device.threads_barrier.wait()
            # Block Logic: Reset the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()

            # Block Logic: Acquire lock to check and update the 'passed' state.
            with self.device.passed_lock:
                # Pre-condition: If the device did not pass the current timepoint,
                # wait for the device-level barrier and reset the 'passed' flag.
                if self.device.passed != True:
                    self.device.device_barrier.wait()
                    self.device.passed = True
                    
                    # Block Logic: Re-add all existing scripts to the workpool for re-processing in the next timepoint.
                    for (script,location) in self.device.scripts:
                        self.device.workpool.put((script,location))
        
        # Block Logic: Stop the thread-specific barrier and the class-level device barrier upon thread termination.
        self.device.threads_barrier.stop_barrier()
        self.device.devices_barrier.stop_barrier()
        # Block Logic: Signal that the timepoint work is complete one last time before exiting.
        self.device.timepoint_done.set()
