


from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem
from collections import deque

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.setup_done = Event() 
        self.script_semaphore = Semaphore(0) 
        self.location_locks = [] 
        self.queue = deque() 
        

        
        self.thread = None
        self.barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def start_thread(self, barrier, locks):
        
        self.thread = DeviceThread(self)
        self.barrier = barrier


        self.location_locks = locks
        self.thread.start()
        self.setup_done.set() 

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            
            barrier = ReusableBarrierSem(len(devices))
            
            locks = []
            for device in devices:
                for location in device.sensor_data:
                    if location not in locks:
                        locks.append((location, Lock()))
            


            for device in devices:
                device.start_thread(barrier, locks)

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            
            self.queue.append(len(self.scripts))
        
        self.script_semaphore.release()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.setup_done.wait()
        self.thread.join()


class WorkerThread(Thread):
    

    def __init__(self, device_thread):
        
        Thread.__init__(self)
        self.device_thread = device_thread

    def run(self):
        
        while True:
            
            self.device_thread.threads_semaphore.acquire()
            
            script = None
            location = None
            if len(self.device_thread.scripts_queue) > 0:
                (script, location) = self.device_thread.scripts_queue.popleft()
            
            if location is None:
                break

            
            lock = next(l for (x, l) in self.device_thread.device.location_locks
                if x == location)

            lock.acquire()
            
            script_data = []
            
            for device in self.device_thread.neighbours:

                data = device.get_data(location)

                if data is not None:
                    script_data.append(data)

            
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                
                for device in self.device_thread.neighbours:
                    device.set_data(location, result)
                
                self.device_thread.device.set_data(location, result)
            
            lock.release()
            
            self.device_thread.worker_semaphore.release()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads_semaphore = Semaphore(0)
        
        self.scripts_queue = deque() 
        self.worker_threads = [] 
        self.neighbours = [] 
        self.worker_semaphore = Semaphore(0)
        
        self.nr_threads = 8 

        
        for _ in xrange(self.nr_threads):
            thread = WorkerThread(self)
            self.worker_threads.append(thread)
            thread.start()

    def run(self):
        
        
        index = 0
        while True:
            
            for _ in xrange(index):
                self.worker_semaphore.acquire()
            
            self.device.barrier.wait()
            
            index = 0
            stop = None
            
            self.neighbours = self.device.supervisor.get_neighbours()
            
            if self.neighbours is None:
                break

            while True:
                
                if not len(self.device.scripts) > index:
                    self.device.script_semaphore.acquire()
                
                if stop is None:
                    if len(self.device.queue) > 0:
                        stop = self.device.queue.popleft()
                
                if stop is not None and stop == index:
                    break
                
                if stop is None and not len(self.device.scripts) > index:
                    continue

                
                (script, location) = self.device.scripts[index]
                
                
                self.scripts_queue.append((script, location))
                self.threads_semaphore.release() 
                
                
                index += 1
        
        for _ in xrange(len(self.worker_threads)):
            self.threads_semaphore.release()
        
        for thread in self.worker_threads:
            thread.join()

