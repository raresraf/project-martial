


from threading import Event, Thread, Condition, Semaphore, Lock, RLock

class ReusableBarrier():
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 

    def wait(self):
        self.cond.acquire()                      
        self.count_threads -= 1;
        if self.count_threads == 0:
            self.cond.notify_all()               
            self.count_threads = self.num_threads    
        else:
            self.cond.wait();                    
        self.cond.release();                     


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id 
        self.sensor_data = sensor_data 
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = [] 
        self.devices = None
        self.timepoint_done = None
        self.semafor = {}
        self.thread = DeviceThread(self) 

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        self.devices = devices
        if self.device_id == 0:
            
            self.timepoint_done = ReusableBarrier(len(self.devices))
            
            for device in self.devices:
                
                for location, data in device.sensor_data.iteritems():
                    
                    if location not in self.semafor:
                        
                        self.semafor.update({location:Lock()})
            for location, data in self.sensor_data.iteritems():
                if location not in self.semafor:
                    
                    self.semafor.update({location:Lock()})
        else:
            for device in self.devices:
                if device.device_id == 0:
                    


                    self.timepoint_done = device.timepoint_done
                    self.semafor = device.semafor
        
        self.thread.start() 

    def assign_script(self, script, location):
        
        if script is not None:
            
            self.scripts.append((script, location))
        else:
            
            self.script_received.set()

    def get_data(self, location):
        
        value = None
        if location in self.sensor_data:
            value = self.sensor_data[location]
        return value

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
                self.device.timepoint_done.wait()
                break

            
            self.device.script_received.wait()
            
            for (script, location) in self.device.scripts:
                
                self.device.semafor[location].acquire()
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
                
                self.device.semafor[location].release()
            
            
            self.device.timepoint_done.wait()
            
            self.device.script_received.clear()
