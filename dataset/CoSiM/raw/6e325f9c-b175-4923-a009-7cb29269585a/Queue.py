




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
        return self.queue.pop()>>>> file: device.py


from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierSem
import time
from Queue import *

class Device(object):
    
    devices_barrier = None
    nr_devices = 0

    def __init__(self, device_id, sensor_data, supervisor):
        

        

        Device.nr_devices += 1
        self.current_neighbours = None
        self.current_neighbours_set = False
        self.access_neighbours = Lock()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.workpool = Queue()
        self.timepoint_done = Event()
        self.threads = []
        self.passed = True
        self.passed_lock = Lock()
        self.device_barrier = None
        self.threads_barrier = ReusableBarrierSem(1)
        self.start_barrier = ReusableBarrierSem(1)
        thread = DeviceThread(self, 1)
        self.threads.append(thread)
        thread.start()

    @classmethod
    def get_devices_barrier(cls):
        
        if cls.devices_barrier == None:
            cls.devices_barrier = ReusableBarrierSem(cls.nr_devices)
        return cls.devices_barrier

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.device_barrier = Device.get_devices_barrier()

    def assign_script(self, script, location):
        
        if script is not None:
            self.workpool.put((script, location))
            self.scripts.append((script, location))

            if len(self.scripts) > len(self.threads) & len(self.threads) < 9:
                self.threads_barrier = ReusableBarrierSem(len(self.threads) + 1)
                thread = DeviceThread(self, len(self.threads) + 1)
                self.threads.append(thread)
                thread.start()
            self.script_received.set()

        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for thread in self.threads:
            thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device, name):
        
        Thread.__init__(self)
        self.device = device
        self.name = "%d" % device.device_id + "%d" % name

    def run(self):
        while True:
            
            with self.device.passed_lock:
                if self.device.passed == True:
                    self.device.passed = False
                    self.device.current_neighbours = self.device.supervisor.get_neighbours()

            if self.device.current_neighbours is None:
                
                break
            else:
                
                self.device.script_received.wait()
                while True:
                    try:
                        (script, location) = self.device.workpool.get_nowait()
                    except Exception:
                        
                        
                        break
                    script_data = []
                    
                    try:
                        for device in self.device.current_neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                    except Exception:
                        break
                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                    if script_data != []:
                        
                        result = script.run(script_data)
                        
                        try:
                            for device in self.device.current_neighbours:
                                device.set_data(location, result)
                        except Exception:
                            break
                        
                        self.device.set_data(location, result)
            
            self.device.timepoint_done.wait()
            self.device.threads_barrier.wait()
            self.device.timepoint_done.clear()

            
            with self.device.passed_lock:
                if self.device.passed != True:
                    self.device.device_barrier.wait()
                    self.device.passed = True
                    
                    for (script,location) in self.device.scripts:
                        self.device.workpool.put((script,location))
        
        self.device.threads_barrier.stop_barrier()
        self.device.devices_barrier.stop_barrier()
        self.device.timepoint_done.set()
