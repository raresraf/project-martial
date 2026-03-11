
"""
This module defines a `Device` class that represents a node in a distributed system or simulation.
Each device runs in its own thread and can execute scripts that operate on sensor data.
Devices are synchronized using a barrier and communicate with their neighbors through a supervisor.
The module also includes custom implementations of Queue, PriorityQueue, and LifoQueue for
inter-thread communication.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
import multiprocessing
from queue import *

class Device(object):
    """
    Represents a device in the simulated environment.

    Each device has a unique ID, its own sensor data, and is managed by a supervisor.
    It executes scripts in a separate thread and synchronizes with other devices at each timepoint.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id: The unique identifier for the device.
            sensor_data: A dictionary representing the device's sensor data.
            supervisor: The supervisor object that manages the device network.
        """
        self.currentScript = 0
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)


        self.thread.start()
        self.barrier = None
        self.queue = Queue()
        self.hash = {}


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def get_unique_id(self, devices):
        """
        Gets the maximum device ID from a list of devices.

        Args:
            devices: A list of Device objects.

        Returns:
            The maximum device ID found in the list.
        """
        max_id = 0;
        for device in devices:
            if (device.device_id > max_id):
                max_id = device.device_id


        return max_id

    def setup_devices(self, devices):
        """
        Sets up the device network, including the synchronization barrier and data locks.

        This method should be called by the device with the highest ID to initialize
        the shared resources for all devices.

        Args:
            devices: A list of all Device objects in the network.
        """

        if (self.device_id == self.get_unique_id(devices)):
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                for k in device.sensor_data:
                    self.hash[k] = Lock()
            
            for device in devices:
                device.barrier = self.barrier
                device.hash = self.hash
        
        pass

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script: The script object to execute.
            location: The location at which the script should be executed.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()


    def get_data(self, location):
        """
        Gets sensor data from a specific location.

        Args:
            location: The location from which to retrieve data.

        Returns:
            The sensor data at the given location, or None if the location does not exist.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data at a specific location.

        Args:
            location: The location at which to set the data.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The thread of execution for a Device.

    This thread manages a pool of worker threads to execute scripts concurrently.
    It waits for scripts to be assigned, executes them, and synchronizes with
    other devices using a barrier.
    """
    def worker(self, q):
        """
        The worker function for the thread pool.

        This function continuously takes scripts from the queue, acquires the necessary locks,
        gathers data from neighboring devices, executes the script, and updates the data.

        Args:
            q: The queue from which to get scripts.
        """
        while True:
            item = q.get()
            my_device = item[0]
            script = item[1]
            location = item[2]
            neighbours = item[3]
            if location is None:
                q.task_done()
                break
            

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
                
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                my_device.set_data(location, result)
            my_device.hash[location].release()
            q.task_done()
            



    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device: The Device object that this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.numCPUs = multiprocessing.cpu_count()


    def run(self):
        """
        The main run loop for the device thread.

        This loop gets neighboring devices from the supervisor, waits for scripts to be
        assigned, and then distributes the scripts to the worker threads for execution.
        After all scripts for a timepoint are done, it waits on the barrier for all
        other devices to finish.
        """
        q = Queue()
        threads = {}
        for i in range(self.numCPUs):
            threads[i] = Thread(target=self.worker, args =(q, ))
            threads[i].daemon = True
            threads[i].start()

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()
            
            for (script, location) in self.device.scripts:
                q.put((self.device, script, location, neighbours))

            q.join()
            self.device.script_received.clear()
            self.device.barrier.wait()
        
        
        for i in range(self.numCPUs):
            q.put((None, None, None, None))
        
        for i in range(self.numCPUs):
            threads[i].join()


from time import time as _time
try:
    import threading as _threading
except ImportError:
    import dummy_threading as _threading
from collections import deque
import heapq

# This section provides a custom implementation of Queue, PriorityQueue, and LifoQueue.
# It appears to be a reimplementation of Python's standard `queue` module, likely for
# compatibility or specific feature requirements within the simulation framework.

__all__ = ['Empty', 'Full', 'Queue', 'PriorityQueue', 'LifoQueue']

class Empty(Exception):
    "Exception raised by Queue.get(block=0)/get_nowait()."
    pass

class Full(Exception):
    "Exception raised by Queue.put(block=0)/put_nowait()."
    pass

class Queue:
    """
    A custom Queue implementation, similar to the standard library's `queue.Queue`.
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
        Indicate that a formerly enqueued task is complete.
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
            while self.unfinished_tasks:
                self.all_tasks_done.wait()
        finally:
            self.all_tasks_done.release()

    def qsize(self):
        """
        Return the approximate size of the queue.
        """
        self.mutex.acquire()
        n = self._qsize()
        self.mutex.release()
        return n

    def empty(self):
        """
        Return True if the queue is empty, False otherwise.
        """
        self.mutex.acquire()
        n = not self._qsize()
        self.mutex.release()
        return n

    def full(self):
        """
        Return True if the queue is full, False otherwise.
        """
        self.mutex.acquire()
        n = 0 < self.maxsize == self._qsize()
        self.mutex.release()
        return n

    def put(self, item, block=True, timeout=None):
        """
        Put an item into the queue.
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
        """
        Put an item into the queue without blocking.
        """
        return self.put(item, False)

    def get(self, block=True, timeout=None):
        """
        Remove and return an item from the queue.
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
        """
        Remove and return an item from the queue without blocking.
        """
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
    A custom PriorityQueue implementation.
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
    A custom LifoQueue (Last-In, First-Out) implementation.
    """

    def _init(self, maxsize):
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop()
