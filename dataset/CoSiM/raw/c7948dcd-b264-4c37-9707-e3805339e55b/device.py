


from threading import Event, Thread, Lock, Semaphore
from reusable_barrier_semaphore import ReusableBarrier
import multiprocessing
import Queue


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.max = 0
        self.supervisor = supervisor
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.root = None
        self.barrier = None
        self.loc_lock = []
        
        for key in self.sensor_data.keys():
            if self.max < key:
                self.max = key

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.devices = devices
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        for dev in devices:
            if dev.device_id == 0:
                self.root = dev
            if self.max < dev.max:
                self.max = dev.max
        if self.device_id == 0:
            for i in range(self.max + 1):
                self.loc_lock.append(Lock())

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

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
        self.queue = []
        self.nr_workers = multiprocessing.cpu_count()
        self.workers = range(self.nr_workers)
        self.neighbours = None
        self.queue_empty = Semaphore(150)
        self.queue_full = Semaphore(0)
        self.end_scripts = Event()
        self.count = 0
        self.count_lock = Lock()

    def run(self):
        for i in range(self.nr_workers):
            self.workers[i] = WorkerThread(self)
        for i in range(self.nr_workers):
            self.workers[i].start()
        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break
            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            if len(self.device.scripts) > 0:
                
                self.count_lock.acquire()
                self.count = len(self.device.scripts)
                self.count_lock.release()
                for (script, location) in self.device.scripts:
                    self.queue_empty.acquire()
                    self.queue.append((script, location))
                    self.queue_full.release()
                
                self.end_scripts.wait()
                self.end_scripts.clear()
            
            self.device.root.barrier.wait()

        
        for i in range(self.nr_workers):
            self.queue_empty.acquire()
            self.queue.append((None, None))
            self.queue_full.release()
        
        for i in range(self.nr_workers):
            self.workers[i].join()


class WorkerThread(Thread):
    

    def __init__(self, master):
        
        Thread.__init__(self)
        self.master = master

    def run(self):
        while True:
            
            self.master.queue_full.acquire()
            (script, location) = self.master.queue.pop()
            self.master.queue_empty.release()
            
            if self.master.neighbours is None:
                break
            else:
                script_data = []
                self.master.device.root.loc_lock[location].acquire()
                
                for device in self.master.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.master.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                if script_data:
                    result = script.run(script_data)
                    self.master.device.set_data(location, result)
                    for device in self.master.neighbours:
                        device.set_data(location, result)
                self.master.device.root.loc_lock[location].release()
                
                self.master.count_lock.acquire()
                self.master.count -= 1
                if self.master.count == 0:
                    


                    self.master.end_scripts.set()
                self.master.count_lock.release()
from threading import *

class ReusableBarrier():
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            


                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
                                                 

class MyThread(Thread):
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",