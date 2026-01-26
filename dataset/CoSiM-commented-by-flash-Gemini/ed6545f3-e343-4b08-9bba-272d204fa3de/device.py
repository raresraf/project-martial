


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
        self.thread.start()
        self.locationlocks = {}
        self.lock = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        bariera = ReusableBarrierSem(len(devices))
        locations = []
        
        
        for dev in devices:
            if (self.device_id == 0):
                dev.bariera = bariera
            for location in dev.sensor_data:
                if not location in locations:
                    locations.append(location)
        
        
        
        
        if (self.device_id == 0):
            for location in locations:
                self.locationlocks[location] = Lock()
            for dev in devices:
                dev.locationlocks = self.locationlocks


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        
        
        
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        
        self.thread.join()


class ScriptThread(Thread):
    
    def __init__(self, device, scripts, locations, neighbours):
        
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts
        self.locations = locations
        self.neighbours = neighbours
    def run(self):
        i = 0
        for script in self.scripts:
            self.device.locationlocks[self.locations[i]].acquire()
            script_data = []
            for device in self.neighbours:
                data = device.get_data(self.locations[i])
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(self.locations[i])
            if data is not None:
                script_data.append(data)
            if script_data != []:
                result = script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.locations[i], result)
                self.device.set_data(self.locations[i], result)
            self.device.locationlocks[self.locations[i]].release()
            i += 1


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        tlist = []
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            
            for index in range(8):
                tlist.append(ScriptThread(self.device, [], [], neighbours))
            index = 0
            
            
            for (script, location) in self.device.scripts:
                tlist[index].scripts.append(script)
                tlist[index].locations.append(location)
                index = (index + 1) % 8
            
            for thread in tlist:
                    thread.start()
            
            for thread in tlist:
                    thread.join()
            
            del tlist[:]
            
            
            self.device.bariera.wait()
