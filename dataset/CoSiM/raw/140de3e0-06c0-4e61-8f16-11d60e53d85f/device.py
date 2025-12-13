


from threading import Event, Thread, Lock, Condition

class ReusableBarrierCond():
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()                  
                                                 
    def wait(self):
        self.cond.acquire()                      
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()               
            self.count_threads = self.num_threads
        else:
            self.cond.wait()                     
                                                 
        self.cond.release()                      

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.timepoint_scripts = []
        
        self.devs_barrier = None
        self.thread = list()
        for i in range(8):
            self.thread.append(DeviceThread(self, i))
        self.dev_barrier = ReusableBarrierCond(len(self.thread))
        
        self.lock_location = Lock()
        
        self.locked_locs = None
        
        
        self.wait_init = Event()      

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            
            size = len(devices)
            barrier = ReusableBarrierCond(8*size)
            self.devs_barrier = barrier

            
            for device in devices:
                if device.devs_barrier is None:
                    if not device.device_id == 0:
                        device.wait_init.set()
                    device.devs_barrier = self.devs_barrier
            
            
            self.locked_locs = dict()
            for device in devices:
                device.locked_locs = self.locked_locs
            for thd in self.thread:


                thd.start()
        else:
            
            self.wait_init.wait()
            for thd in self.thread:
               thd.start()
        


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
        
        for thd in self.thread:
            thd.join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        
        while True:
			
            self.device.devs_barrier.wait()

            if self.thread_id is 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            
            else:
                pass



            self.device.dev_barrier.wait()
            neighbours = self.device.neighbours
            if neighbours is None:
                break           
            
            
            self.device.timepoint_done.wait()
            if self.thread_id is 0:
                self.device.timepoint_scripts = [i for i in self.device.scripts]
            self.device.dev_barrier.wait()

            
            for (script, location) in self.device.timepoint_scripts:
				
                self.device.lock_location.acquire()
                if not self.device.timepoint_scripts:
                    self.device.lock_location.release()
                    break
                (script, location) = self.device.timepoint_scripts.pop()
                
                if self.device.locked_locs.has_key(location):
                    pass
                else:


                    self.device.locked_locs[location] = Lock()
				
                self.device.locked_locs[location].acquire()
                self.device.lock_location.release()

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
                
                self.device.locked_locs[location].release()

            
            self.device.dev_barrier.wait()
            self.device.timepoint_done.clear()
