


from threading import Thread,Event,Condition,Lock

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
        
        self.lock = None
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        for i in devices:


	        if self.device_id == 0:
	            Device.barrier = ReusableBarrier(len(devices))
        self.lock = Lock()
        self.thread = DeviceThread(self, Device.barrier , self.lock)
        self.thread.start()


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
    

    def __init__(self, device , barrier , lock):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.barrier = barrier
        self.lock = lock
        
    def run(self):
        
        while True:
            
            self.barrier.wait()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            
            self.device.timepoint_done.wait()
            
            self.device.timepoint_done.clear()

            
            self.lock.acquire()
            for (script, location) in self.device.scripts:
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
            self.lock.release()