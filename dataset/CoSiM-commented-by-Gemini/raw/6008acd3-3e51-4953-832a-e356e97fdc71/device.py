


from threading import Event, Thread, Lock
import barrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.my_lock = None
        self.timepoint_done = Event()
        self.bariera = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            lent = len(devices)
            bariera = barrier.ReusableBarrier(lent)


            my_lock = Lock()
            for device in devices:
                device.bariera = bariera
            for device in devices:
                device.my_lock = my_lock

        
        

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
            
            for (script, location) in self.device.scripts:
                self.device.my_lock.acquire()
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
                self.device.my_lock.release()
	    
            self.device.timepoint_done.clear()
            self.device.bariera.wait()
