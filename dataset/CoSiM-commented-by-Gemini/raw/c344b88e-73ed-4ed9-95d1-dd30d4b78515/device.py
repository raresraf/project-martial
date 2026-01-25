


from threading import Event, Thread, RLock, Lock, Semaphore


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
        self.scripts = []
        self.last_script = Event()
        self.thread = DeviceThread(self)
        self.timepoint_done = None
        self.loc_lock = None
        self.thread.start()
        
    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        if self.timepoint_done is None:
            barrier = ReusableBarrier(len(devices))
            dic = {}
            for dev in devices:
                dev.timepoint_done = barrier
                dev.loc_lock = dic

    def assign_script(self, script, location):
        
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.last_script.set()

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
    
    
    
    
    def run_script(self, lock, neighbours, location, script):
        

        lock.acquire();
        if not (self.device.loc_lock).has_key(location):
            self.device.loc_lock[location] = Lock()
        lock.release()
        
        


        self.device.loc_lock.get(location).acquire()
        script_data = []
        
        for dev in neighbours:
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)
               
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
                
        if script_data != []:
            
            result = script.run(script_data)
            
            
            for dev in neighbours:
                dev.set_data(location, result)
                
                self.device.set_data(location, result)
        
        (self.device.loc_lock.get(location)).release()
                

    def run(self):
    
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            
            self.device.last_script.wait()
            
            
            
            lock = RLock()
            threads = []

            for (script, location) in self.device.scripts:
                
               	thread = Thread(target = self.run_script, args = (lock, neighbours, location, script))
                thread.start()
                threads.append(thread)
            
            for thread in threads:
                thread.join()
 
            
            
            self.device.timepoint_done.wait()
            self.device.last_script.clear()
