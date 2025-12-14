


from threading import Thread, Lock
from Queue import Queue
from utils import SharedDeviceData
from utils import ThreadPool

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.num_cores = 8  
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.new_scripts = Queue()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == 0:
            
            
            shared_data = SharedDeviceData(len(devices))
            
            for data in self.sensor_data:
                if data not in shared_data.location_locks:
                    shared_data.location_locks[data] = Lock()

            for dev in devices:
                dev.shared_data = shared_data

        

    def assign_script(self, script, location):
        

        
        self.new_scripts.put((script, location))

    def get_data(self, location):
        

        
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        

        
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        
        

        
        thread_pool = ThreadPool(self.device.num_cores)
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            for (script, location) in self.device.scripts:
                thread_pool.submit(RunScript(script, location, neighbours,
                                             self.device))

            
            while True:
                (script, location) = self.device.new_scripts.get()
                if script is None: 
                    break

                
                self.device.shared_data.ll_lock.acquire()
                if location not in self.device.shared_data.location_locks:
                    self.device.shared_data.location_locks[location] = Lock()
                self.device.shared_data.ll_lock.release()

                thread_pool.submit(RunScript(script, location, neighbours,
                                             self.device))
                self.device.scripts.append((script, location))

            thread_pool.shutdown() 
            thread_pool.wait_termination(False) 

            
            self.device.shared_data.timepoint_barrier.wait()

        thread_pool.wait_termination() 

class RunScript(object):
    

    def __init__(self, script, location, neighbours, device):
        

        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        

        
        
        
        
        self.device.shared_data.ll_lock.acquire()
        lock = self.device.shared_data.location_locks[self.location]
        self.device.shared_data.ll_lock.release()

        
        script_data = []

        lock.acquire()  

        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        lock.release() 



from threading import Condition, Semaphore, Event, Lock
from threading import Thread
from Queue import Queue

class CyclicBarrier(object):
    

    def __init__(self, parties):
        

        self.parties = parties
        self.count = 0
        self.condition = Condition()

    def wait(self):
        

        self.condition.acquire()
        self.count += 1
        if self.count == self.parties:
            self.condition.notifyAll() 
            self.count = 0  
        else:
            self.condition.wait()

        self.condition.release()

class ThreadPool(object):
    

    def __init__(self, num_threads):
        
        self.num_threads = num_threads

        self.task_queue = Queue() 
        self.num_tasks = Semaphore(0) 
        self.stop_signal = Event() 
        self.shutdown_signal = Event()

        self.threads = []
        for i in xrange(0, num_threads):
            self.threads.append(Worker(self.task_queue,
                                       self.num_tasks,
                                       self.stop_signal))

        
        for i in xrange(0, num_threads):
            self.threads[i].start()

    def submit(self, task):
        
        if self.shutdown_signal.is_set():
            return 

        self.task_queue.put(task)
        self.num_tasks.release()

    def shutdown(self):
        
        self.shutdown_signal.set()

    def wait_termination(self, end=True):
        
        self.task_queue.join()
        if end is True:
            self.stop_signal.set() 
            for i in xrange(0, self.num_threads):
                self.task_queue.put(None) 
                self.num_tasks.release()

            for i in xrange(0, self.num_threads):
                self.threads[i].join()
        else:
            self.shutdown_signal.clear()


class Worker(Thread):
    

    def __init__(self, task_queue, num_tasks, stop_signal):
        
        Thread.__init__(self)
        self.task_queue = task_queue
        self.num_tasks = num_tasks
        self.stop_signal = stop_signal

    def run(self):
        
        while True:
            
            self.num_tasks.acquire()
            if self.stop_signal.is_set():
                break

            
            task = self.task_queue.get()

            
            task.run()
            self.task_queue.task_done()

class SharedDeviceData(object):
    

    def __init__(self, num_devices):
        

        self.num_devices = num_devices
        self.timepoint_barrier = CyclicBarrier(num_devices)

        
        
        
        self.location_locks = {}

        
        self.ll_lock = Lock()
