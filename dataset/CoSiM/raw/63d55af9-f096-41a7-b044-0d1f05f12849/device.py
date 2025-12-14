


import barrier
from threading import Event, Thread, Lock


class Device(object):
    
    
    barrier = None
    lock = None

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if(self.device_id == 0):


             Device.barrier = barrier.ReusableBarrierCond(len(devices))
             Device.lock = Lock()
        
        self.thread = DeviceThread(self, Device.barrier, Device.lock)
        self.thread.start()        
        

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
    

    def __init__(self, device, barrier, lock):
        
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

            self.device.script_received.wait()
            self.device.script_received.clear()

            
            for (script, location) in self.device.scripts:
                script_data = []
                
                for device in neighbours:
                    self.lock.acquire()
                    data = device.get_data(location)
                    self.lock.release()
                    if data is not None:
                        script_data.append(data)
                
                self.lock.acquire()
                data = self.device.get_data(location)
                self.lock.release()
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in neighbours:
                        self.lock.acquire()
                        device.set_data(location, result)
                        self.lock.release()
                    
                    self.lock.acquire()
                    self.device.set_data(location, result)
                    self.lock.release()

            
            self.device.timepoint_done.wait()
