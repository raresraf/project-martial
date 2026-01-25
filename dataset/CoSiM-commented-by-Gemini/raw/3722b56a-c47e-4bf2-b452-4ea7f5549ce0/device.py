


from threading import Event, Thread
from threading import Semaphore, Lock

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
                
                for i in range(self.num_threads):
                    
                    
                    threads_sem.release()
                
                count_threads[0] = self.num_threads
        
        threads_sem.acquire()
        



class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        

        self.barrier = None 
        self.InitializationEvent = Event() 
        self.LockLocation = None 
        self.LockDict = Lock() 

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            
            

            n = len(devices)
            self.barrier = ReusableBarrier(n)   
            self.LockLocation = {}  

            for idx in range(len(devices)):
                d = devices[idx]


                d.LockLocation = self.LockLocation
                d.barrier = self.barrier
                if d.device_id == 0:
                    pass
                else:
                    d.InitializationEvent.set()
        else:
            self.InitializationEvent.wait()

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
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        while True:
            
            self.device.barrier.wait()

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()


            dev_scripts = self.device.scripts

            
            for (script, location) in self.device.scripts:
                self.device.LockDict.acquire()

                if location not in self.device.LockLocation.keys():
                    self.device.LockLocation[location] = Lock()



                self.device.LockLocation[location].acquire()

                self.device.LockDict.release()

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

                self.device.LockLocation[location].release()

            
            self.device.timepoint_done.clear()
from threading import Semaphore, Lock

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
                
                for i in range(self.num_threads):
                    
                    
                    threads_sem.release()
                
                count_threads[0] = self.num_threads
        
        threads_sem.acquire()
        
