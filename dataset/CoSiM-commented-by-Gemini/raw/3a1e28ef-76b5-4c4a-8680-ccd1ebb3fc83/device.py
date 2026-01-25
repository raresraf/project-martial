


from threading import *
from barrier import *


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
        self.lock = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        DeviceThread.barr.set_th(len(devices))


    def assign_script(self, script, location):
        
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()
    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    
    barr = ReusableBarrierCond()

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            DeviceThread.barr.wait()
            
            self.device.timepoint_done.wait()
            
            for (script, location) in self.device.scripts:
                script_data = []
                
                for device in neighbours:
                    device.lock.acquire()
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                if script_data != []:
                    
                    result = script.run(script_data)
                    
                    for device in neighbours:
                        device.lock.release()
                        device.set_data(location, result)

                    
                    self.device.set_data(location, result)
            
            
            self.device.timepoint_done.clear()
