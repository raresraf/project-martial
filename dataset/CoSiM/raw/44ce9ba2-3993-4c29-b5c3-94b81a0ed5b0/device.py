


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barrier = None
        self.thread.start()
        self.block_location = None
    def __str__(self):
        
        return "Device %d" % self.device_id
    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            
            
            
            self.barrier = ReusableBarrierSem(len(devices))
            locations = []
            
            
            
            
            for device in devices:
                for location in device.sensor_data:
                    if location is not None:
                        if location not in locations:
                            locations.append(location)
            self.block_location = []
            for _ in xrange(len(locations)):
                self.block_location.append(Lock())
            for device in devices:
                device.barrier = self.barrier
                device.block_location = self.block_location
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
class OneThread(Thread):
    
    def __init__(self, myid, device, location, neighbours, script):
        Thread.__init__(self)
        self.myid = myid


        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script
    def run(self):
        with self.device.block_location[self.location]:
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

            
            threads = []
            myid = 0
            for (script, location) in self.device.scripts:
                thread = OneThread(myid, self.device, location, neighbours, script)
                threads.append(thread)
                thread.start()
                myid += 1
            for thread in threads:
                thread.join()
            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
