


from threading import Event, Thread, Lock, Semaphore
class ReusableBarrier(object):	
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
                for _ in range(self.num_threads):
                    threads_sem.release()
                    
                count_threads[0] = self.num_threads        
        threads_sem.acquire()        
                                    
class MyThread(Thread):	
    def __init__(self, neighbours, script, location, device):
        Thread.__init__(self)
        self.neighbours = neighbours


        self.script = script
        self.location = location
        self.device = device
        self.script_data = []
    def run(self):
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                self.script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            self.script_data.append(data)
        if self.script_data != []:
            
            result = self.script.run(self.script_data)
            
            for device in self.neighbours:
                device.set_data(self.location, result)
                
            self.device.set_data(self.location, result)
        self.script_data = []
class Device(object):
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
    def __str__(self):
        
        return "Device %d" % self.device_id
    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for i in xrange(len(devices)):
                devices[i].barrier = barrier
    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
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

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            freds = []
            for (script, location) in self.device.scripts:
                fred = MyThread(neighbours, script, location, self.device)
                freds.append(fred)
                
            for i in freds:
                i.start()
            for i in freds:
                i.join()
            
            self.device.timepoint_done.clear()
            self.device.barrier.wait() 
