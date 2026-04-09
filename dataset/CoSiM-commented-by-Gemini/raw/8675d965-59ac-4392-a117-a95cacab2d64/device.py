"""
This module provides a framework for simulating a network of distributed devices.

It defines the `Device` class, which represents a node in the network, and a
`DeviceThread` class that manages the execution logic for each device. The
simulation uses a barrier to synchronize all devices at discrete time steps and
location-based locks to ensure data consistency when multiple devices interact
with the same data point.

Note: This file includes a vendored-in copy of Python's standard `queue` module
at the end.
"""




from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
import multiprocessing
from queue import *

class Device(object):
    """
    Represents a single device (or node) in the simulated network.

    Each device holds its own sensor data, can be assigned scripts to execute,
    and communicates with neighboring devices. It uses a master `DeviceThread`
    to manage its lifecycle and operations.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary mapping locations to sensor values.
        supervisor: An external object responsible for orchestrating the network.
        scripts (list): A list of (script, location) tuples to be executed.
        barrier (ReusableBarrierCond): A shared barrier for time-step synchronization.
        hash (dict): A shared dictionary mapping locations to Lock objects for
                     ensuring data consistency.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor: The supervisor object managing the simulation.
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
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def get_unique_id(self, devices):
        """Finds the highest device ID in a list of devices."""
        max_id = 0;
        for device in devices:
            if (device.device_id > max_id):
                max_id = device.device_id
        return max_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for all devices in the simulation.

        A single device (elected by having the highest ID) creates a shared
        synchronization barrier and a shared dictionary of locks. These shared
        resources are then distributed to all other devices to ensure that the
        entire network operates in synchronized time steps and with data consistency.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """

        # "Leader election": the device with the highest ID sets up shared resources.
        if (self.device_id == self.get_unique_id(devices)):
            # Create a barrier to synchronize all devices at each time step.
            self.barrier = ReusableBarrierCond(len(devices)) 
            # Create a shared mapping of locations to locks for data consistency.
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
        Assigns a script to be executed by this device.

        The supervisor calls this method to give work to the device for the
        current time step.

        Args:
            script: The script object to be executed.
            location: The location context for the script execution.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for this time step have been assigned.
            self.script_received.set() 


    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device.

        Args:
            location: The location to query for data.

        Returns:
            The data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location on this device.

        Args:
            location: The location to update.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a single Device.

    This thread manages a pool of worker sub-threads to process scripts in
    parallel. It coordinates with other devices using a shared barrier to
    ensure synchronized time steps across the entire simulation.
    """
    
    def worker(self, q):
        """
        The target function for the internal worker thread pool.

        Pulls tasks from a queue. Each task involves executing a script for a
        specific location. It ensures data consistency by acquiring a shared,
        location-specific lock before accessing or modifying data.

        Args:
            q (Queue): The queue from which to get script execution tasks.
        """
        while True:
            item = q.get()
            my_device = item[0]
            script = item[1]
            location = item[2]
            neighbours = item[3]
            # A None location is a sentinel value to terminate the worker.
            if location is None:
                q.task_done()
                break
            
            # Acquire the global lock for this specific location to ensure
            # that no other device is operating on this location's data.
            my_device.hash[location].acquire()
            script_data = []
            
            # Gather data from all neighboring devices for the given location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gather this device's own data for the location.
            data = my_device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                # Execute the script with the aggregated data.
                result = script.run(script_data)
                # Broadcast the result by setting the data on all neighbors and itself.
                for device in neighbours:
                    device.set_data(location, result)
                my_device.set_data(location, result)

            # Release the lock for the location.
            my_device.hash[location].release()
            q.task_done()
            



    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.numCPUs = multiprocessing.cpu_count()


    def run(self):
        """
        The main event loop for the device.

        This loop represents the progression of time in the simulation. In each
        iteration (timepoint), the device waits for scripts, executes them using
        an internal thread pool, and then waits at a global barrier for all
        other devices to finish before proceeding to the next timepoint.
        """
        q = Queue()
        threads = {}
        # Create an internal thread pool to process script tasks in parallel.
        for i in range(self.numCPUs):
            threads[i] = Thread(target=self.worker, args =(q, ))
            threads[i].daemon = True
            threads[i].start()

        # Main simulation loop (timepoint progression).
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            # If the supervisor returns no neighbours, the simulation is over.
            if neighbours is None:
                break

            # 1. Wait for the supervisor to assign all scripts for this timepoint.
            self.device.script_received.wait()
            
            # 2. Enqueue all assigned scripts for the internal workers to process.
            for (script, location) in self.device.scripts:
                q.put((self.device, script, location, neighbours)) 

            # 3. Wait for the internal workers to finish all tasks for this timepoint.
            q.join()            
            self.device.script_received.clear()

            # 4. Synchronize with all other devices. No device proceeds until all
            #    have reached this barrier. This marks the end of a timepoint.
            self.device.barrier.wait()
        
        
        # Shutdown signal: send sentinel values to terminate worker threads.
        for i in range(self.numCPUs):
            q.put((None, None, None, None))
        
        for i in range(self.numCPUs):
            threads[i].join()



# ---------------------------------------------------------------------------
# The following is a vendored-in copy of Python's standard `queue` module.
# It is included here to make the project self-contained.
# ---------------------------------------------------------------------------

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
