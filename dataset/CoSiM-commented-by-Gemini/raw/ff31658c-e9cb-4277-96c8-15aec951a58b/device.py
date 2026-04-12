"""
This module defines the core components for a distributed device simulation.

It includes the `Device` class, which represents a single node in the system,
and the `DeviceThread` class, which manages the execution logic for a device,
including running scripts and synchronizing with other devices.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
import multiprocessing
from queue import *

class Device(object):
    """
    Represents a single device in a simulated distributed network.

    Each device has a unique ID, local sensor data, and can execute scripts.
    It communicates with a central supervisor and synchronizes with other devices
    using a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor data.
            supervisor: The central supervisor object that manages the simulation.
        """
        self.currentScript = 0
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been received.
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # The main execution thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Barrier for synchronizing all devices at the end of a timepoint.
        self.barrier = None
        # Queue for worker threads to process script tasks.
        self.queue = Queue()
        # A dictionary of locks to protect access to sensor data locations.
        self.hash = {}


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def get_unique_id(self, devices):
        """Finds the maximum device ID in a list of devices."""
        max_id = 0
        for device in devices:
            if (device.device_id > max_id):
                max_id = device.device_id
        return max_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for all devices in the simulation.

        This method should be called on a single "leader" device. It creates a
        shared barrier and a set of locks for all sensor data locations, then
        distributes these shared objects to all other devices.
        """
        # Block Logic: Only the device with the highest ID initializes the shared resources.
        if (self.device_id == self.get_unique_id(devices)):
            self.barrier = ReusableBarrierCond(len(devices))
            # Create a lock for each unique sensor data location across all devices.
            for device in devices:
                for k in device.sensor_data:
                    self.hash[k] = Lock()
            
            # Distribute the shared barrier and locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.hash = self.hash
        
        pass

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        If the script is None, it signals that all scripts for the current
        timepoint have been assigned.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that the device has received all scripts for this step.
            self.script_received.set() 


    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the execution logic for a Device using a pool of worker threads.
    """
    def worker(self, q):
        """
        The target function for worker threads.

        Pulls script execution tasks from a queue, processes them, and updates
        sensor data in a thread-safe manner.
        """
        while True:
            item = q.get()
            my_device = item[0]
            script = item[1]
            location = item[2]
            neighbours = item[3]
            # A `None` location is a sentinel value to terminate the worker.
            if location is None:
                q.task_done()
                break
            
            # Acquire a lock for the specific data location to prevent race conditions.
            my_device.hash[location].acquire()
            script_data = []
            
            # Gather data from neighboring devices at the same location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Include the device's own data.
            data = my_device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: Only run the script if there is data to process.
            if script_data != []:
                # Execute the script and update the data on all relevant devices.
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                my_device.set_data(location, result)
            my_device.hash[location].release()
            q.task_done()

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.numCPUs = multiprocessing.cpu_count()


    def run(self):
        """
        The main loop for the device thread.

        It creates a worker thread pool, then enters a loop to wait for and
        execute scripts for each timepoint, synchronizing with other devices
        using a barrier.
        """
        q = Queue()
        threads = {}
        # Create a pool of worker threads.
        for i in range(self.numCPUs):
            threads[i] = Thread(target=self.worker, args =(q, ))
            threads[i].daemon = True
            threads[i].start()

        # Invariant: The main loop continues as long as the supervisor provides neighbors.
        while True:
            # Get the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait until all scripts for the current timepoint are assigned.
            self.device.script_received.wait()
            
            # Put all assigned script tasks into the queue for the workers.
            for (script, location) in self.device.scripts:
                q.put((self.device, script, location, neighbours)) 

            # Wait for all tasks in the queue to be completed.
            q.join()            
            self.device.script_received.clear()
            # Synchronize with all other devices before proceeding to the next timepoint.
            self.device.barrier.wait()
        
        # Shutdown sequence: send sentinel values to terminate worker threads.
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

__all__ = ['Empty', 'Full', 'Queue', 'PriorityQueue', 'LifoQueue']

class Empty(Exception):
    "Exception raised by Queue.get(block=0)/get_nowait()."
    pass

class Full(Exception):
    "Exception raised by Queue.put(block=0)/put_nowait()."
    pass

class Queue:
    
    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self._init(maxsize)
        
        
        
        
        self.mutex = _threading.Lock()
        
        
        self.not_empty = _threading.Condition(self.mutex)
        
        
        self.not_full = _threading.Condition(self.mutex)
        
        
        self.all_tasks_done = _threading.Condition(self.mutex)
        self.unfinished_tasks = 0

    def task_done(self):
        
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
    

    def _init(self, maxsize):
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item, heappush=heapq.heappush):
        heappush(self.queue, item)

    def _get(self, heappop=heapq.heappop):
        return heappop(self.queue)


class LifoQueue(Queue):
    

    def _init(self, maxsize):
        self.queue = []

    def _qsize(self, len=len):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop()
