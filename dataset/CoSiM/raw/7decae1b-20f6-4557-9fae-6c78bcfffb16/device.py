


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.lock_for_data = [None] * 100
        self.inside_threads = [] 
        self.stored_devices = [] 
        self.barrier = None 

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        for i in range(100):
            self.lock_for_data[i] = Lock()
        
        barrier = ReusableBarrierSem(len(devices))
        


        for device in devices:
            
            device.barrier = barrier
            
            self.stored_devices.append(device)

    def assign_script(self, script, location):
        
        if script is not None:
            
            self.scripts.append((script, location))
            


            for device in self.stored_devices:
                
                self.lock_for_data = device.lock_for_data
            
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
                
                inside_thread = InsideDeviceThread(self.device, script, location, neighbours)
                
                self.device.inside_threads.append(inside_thread)
                
                inside_thread.start()

            for inside_thread in self.device.inside_threads:
                inside_thread.join()

            
            del self.device.inside_threads[:]
            
            self.device.timepoint_done.clear()
            
            self.device.barrier.wait()


class InsideDeviceThread(Thread):
    

    def __init__(self, device, script, location, neighbours):
        
        Thread.__init__(self, name="Inside Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours


    def run(self):
        
        self.device.lock_for_data[self.location].acquire()

        self.device.script_received.wait()

        script_data = []

        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        
        self.device.lock_for_data[self.location].release()
