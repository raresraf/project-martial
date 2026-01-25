


from threading import Event, Thread, Lock, Semaphore


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



class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        
        self.barrier = None
        
        self.lock_location = None
        
        self.lock_script = Lock()
        
        self.lock_neighbours = Lock()
        
        self.available = []
        
        self.neighbours = None
        
        self.init_done = Event()
        
        self.update_neighbours = True 
        
        self.threads = []


        for i in range(8):
            self.threads.append(DeviceThread(self))
            self.threads[i].start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        


        if self.device_id == 0:

            self.barrier = ReusableBarrierSem(len(devices))
            self.lock_location = []

            for _ in range(200):
                self.lock_location.append(Lock())

            self.init_done.set()
        else:
            for device in devices:
                if device.device_id == 0:
                    
                    device.init_done.wait()
                    
                    self.barrier = device.barrier
                    self.lock_location = device.lock_location
                    return

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            
            self.available.append(True)
        else:
            
            self.barrier.wait()
            
            self.reset()
            self.timepoint_done.set()
            

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for thread in self.threads:
            if thread.isAlive():
                thread.join()

    def reset(self):
        
        self.lock_neighbours.acquire()
        self.update_neighbours = True
        self.lock_neighbours.release()
        for i in range(len(self.available)):
            self.available[i] = True


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:

            
            self.device.lock_neighbours.acquire()

            if self.device.update_neighbours:


                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.update_neighbours = False

            self.device.lock_neighbours.release()

            neighbours = self.device.neighbours

            if neighbours is None:
                break

            for (script, location) in self.device.scripts:

                
                self.device.lock_script.acquire()

                index = self.device.scripts.index((script, location))
                if self.device.available[index]:
                    self.device.available[index] = False
                else:
                    self.device.lock_script.release()
                    continue



                self.device.lock_script.release()

                
                self.device.lock_location[location].acquire()

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

                self.device.lock_location[location].release()

            self.device.timepoint_done.wait()
