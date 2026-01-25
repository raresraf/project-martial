

 
from threading import *
 
 
class ReusableBarrierSem():
    
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        self.phase1()
        self.phase2()
    
    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        
        self.threads_sem1.acquire()
    
    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        
        self.threads_sem2.acquire()




from threading import Event, Thread

 
class Device(object):
    
 
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self, 0)
     
 
        self.com_barrier = None
        self.initialize = Event()
        self.locked_locations = None
        self.lock_dict = Lock()
 
    def __str__(self):
        
        return "Device %d" % self.device_id
 
    def setup_devices(self, devices):
        
        
        if self.device_id != 0:
            self.initialize.wait()
        else:
            self.com_barrier = ReusableBarrierSem(len(devices))
            self.locked_locations = {}
            for d in devices:


                d.com_barrier = self.com_barrier
                d.locked_locations = self.locked_locations
                if (d.device_id == 0):
                    pass
                else:
                    d.initialize.set()
            
        self.thread.start()    
 
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
    
 
    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id
 
    def run(self):
        
        while True:
            
            self.device.com_barrier.wait()

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break



            self.device.timepoint_done.wait()
 
            
            for (script, location) in self.device.scripts:
                
                self.device.lock_dict.acquire()
                
                if (location not in self.device.locked_locations):
                    self.device.locked_locations[location] = Lock()
                else:
                    pass



                self.device.locked_locations[location].acquire();
                self.device.lock_dict.release()
                
                script_data = []
                
 
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
 
                if script_data != []:
                    
                    result = script.run(script_data)
 
                    
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
 
            
                self.device.locked_locations[location].release();
            
            self.device.timepoint_done.clear()
