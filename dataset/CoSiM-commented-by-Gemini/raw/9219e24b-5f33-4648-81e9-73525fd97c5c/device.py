


from threading import Event, Thread, Condition, Lock


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barr = None
        self.lock = None
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
		
        if devices[0].barr is None and self.device_id == devices[0].device_id:
                barr = CondBarrier(len(devices))
                for i in devices:
                        i.barr = barr
        lock = Lock()	
        for d in devices:
                d.lock = lock 
		
		

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
	
    def shutdown(self):
        
        self.thread.join()

class CondBarrier():
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

            
            
            for (script, location) in self.device.scripts:
                self.device.lock.acquire()
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
                self.device.lock.release()
            
            self.device.timepoint_done.clear()
            self.device.barr.wait()
