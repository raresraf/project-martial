


from threading import Semaphore, Lock, Event, Thread

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
        self.timepoint_done = Event()


        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        
        self.hash_ld = {}

    def __str__(self):
        
        return "Device %d" % self.device_id

    
    def exists(self,key):
        nr = 0;
        for k in self.hash_ld.keys():
            if(k == key):
                nr += 1


        return nr
    def setup_devices(self, devices):
        
        
        nrd = len(devices)
        if(self.device_id == 0):
            
            self.barrier = ReusableBarrier(nrd)
            
            
            for device in devices:
                if(device.device_id != 0):
                    device.barrier = self.barrier
                    device.hash_ld = self.hash_ld

        
        for device in devices:
            for k in device.sensor_data.keys():
                if(self.exists(k) == 0):
                    self.hash_ld[k] = Lock()

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
        
        self.t = []

    
    def method(self,script, script_data):
        result = script.run(script_data)
        return result

    
    def method_scripts(self,neighbours,script,location):
        
        
         self.device.hash_ld[location].acquire()
         script_data = []
         for device in neighbours:
             data = device.get_data(location)
             if data is not None:
                 script_data.append(data)
         data = self.device.get_data(location)
         if data is not None:
             script_data.append(data)

         if script_data != []:
             result = self.method(script,script_data)
             for device in neighbours:
                 device.set_data(location, result)
             self.device.set_data(location, result)
         self.device.hash_ld[location].release()

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            self.t = []
            number = len(self.device.scripts)
            number_of_threads = min(8, number)
            nr = 1
            for (script, location) in self.device.scripts:
                
                
                self.t.append(Thread(target = self.method_scripts, args = (neighbours,script,location)))
                if(nr == number_of_threads):
                    for i in range(0,nr):
                        self.t[i].start()
                    for i in range(0,nr):
                        self.t[i].join()
                    self.t = []
                    nr = 0
                nr += 1

            
            
            self.device.barrier.wait()
