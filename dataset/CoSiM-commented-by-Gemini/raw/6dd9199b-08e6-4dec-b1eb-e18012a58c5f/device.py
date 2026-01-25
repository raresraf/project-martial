


from threading import Event, Thread, Lock
from barrier import ReusableBarrier

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

        self.barrier1 = None
        self.barrier2 = None

        self.location_lock = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            self.barrier1 = ReusableBarrier(len(devices))
            self.barrier2 = ReusableBarrier(len(devices))

            for device in devices:
                device.barrier1 = self.barrier1
                device.barrier2 = self.barrier2

            
            max_loc = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > max_loc:
                        max_loc = location
            while max_loc >= 0:
                self.location_lock.append(Lock())
                max_loc = max_loc - 1
            
             
              
            for device in devices:
                device.location_lock = self.location_lock


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            
        else:
            self.script_received.set()
            self.barrier2.wait()

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
            self.device.script_received.wait()
            self.device.barrier2.wait()
            self.device.barrier1.wait()

            if neighbours is None:
                break

            

            
            for (script, location) in self.device.scripts:
                self.device.location_lock[location].acquire()

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

            
                self.device.location_lock[location].release()
