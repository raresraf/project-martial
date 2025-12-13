


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
        self.barrier_is_up = Event() 

        
        self.location_acces = {}
        self.device0 = None 

        
        self.can_receive_scripts = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            self.barrier_is_up.set()

        
        for device in devices:
            if device.device_id == 0:
                self.device0 = device
    def assign_script(self, script, location):
        

        


        self.can_receive_scripts.acquire()
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
        self.can_receive_scripts.release()

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
        self.core_semaphore = Semaphore(8) 

    def run(self):
        timepoint = 0
        executor_service = ScriptExecutorService()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            
            self.device.can_receive_scripts.acquire()

            
            self.device.timepoint_done.clear()

            
            if timepoint == 0:
                self.device.device0.barrier_is_up.wait()
                timepoint = timepoint + 1

            for (script, location) in self.device.scripts:
                executor_service.submit_job(script, self.device, location, neighbours)
            
            executor_service.wait_finish()

            
            self.device.can_receive_scripts.release()

            
            
            self.device.device0.barrier.wait()

class ScriptExecutorService(object):
    
    def __init__(self):
        
        self.core_semaphore = Semaphore(8) 
        self.executors = [] 
    def submit_job(self, script, device, location, neighbours):
        

        
        executor = ScriptExecutor(script, device, location, neighbours, self.core_semaphore)
        
        self.core_semaphore.acquire()
        executor.start()
        self.executors.append(executor)

    def wait_finish(self):
        
        for executor in self.executors:
            executor.join()

class ScriptExecutor(Thread):
    
    def __init__(self, script, device, location, neighbours, core_semaphore):
        
        Thread.__init__(self, name="Script Executor pentru device-ul %d" % device.device_id)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script
        self.core_semaphore = core_semaphore

    def run(self):

        
        if self.location not in self.device.device0.location_acces:
            self.device.device0.location_acces[self.location] = Lock()


        
        self.device.device0.location_acces[self.location].acquire()
        script_data = []
        data = None

        
        if self.neighbours is not None:
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            if self.neighbours is not None:
                for device in self.neighbours:
                    device.set_data(self.location, result)

            self.device.set_data(self.location, result)

        
        self.device.device0.location_acces[self.location].release()

        
        self.core_semaphore.release()
        