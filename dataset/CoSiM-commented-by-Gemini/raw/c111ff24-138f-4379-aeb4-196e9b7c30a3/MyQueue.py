

from Queue import Queue 
from threading import Thread

class MyQueue():
    
    def __init__(self, num_threads):
        
        self.queue = Queue(num_threads)
        self.threads = []
        self.device = None

        


        for _ in xrange(num_threads):
            thread = Thread(target=self.run)
            self.threads.append(thread)
        
        
        for thread in self.threads:
            thread.start()
    
    def run(self):
        
        while True:
            
            neighbours, script, location = self.queue.get()

            
            if neighbours is None and script is None:
                self.queue.task_done()
                return
        
            script_data = []
            
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)
                
                
                for device in neighbours:


                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                
                self.device.set_data(location, result)
            
            self.queue.task_done()
    
    def finish(self):
        
        
        self.queue.join()

        
        for _ in xrange(len(self.threads)):
           self.queue.put((None, None, None))

        
        for thread in self.threads:
            thread.join()


from threading import Thread, Event, Lock, Semaphore
from MyQueue import MyQueue

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

class Device(object):
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.location_locks = {location: Lock() for location in self.sensor_data}
        self.scripts_available = False
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        
        
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]     
        else:
            return None

    def set_data(self, location, data):
        
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
        else:
            return None

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = MyQueue(8)

    def run(self):
        
        
        self.queue.device = self.device
        while True:
            
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            while True:
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    if self.device.scripts_available:
                        self.device.scripts_available = False

                        
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
            
                    else:
            
                        
                        self.device.timepoint_done.clear()
                        self.device.scripts_available = True
                        break
            
            
            self.queue.queue.join()
            self.device.barrier.wait()

        
        self.queue.finish()
