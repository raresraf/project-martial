


from threading import Event, Thread, Lock, Condition
import barrier

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
        
        
        self.bariera = barrier.ReusableBarrierCond(1)
        
        self.data_lock = Lock()
        
        self.script_lock = Lock()
        
        
        self.locationcondition = Condition()
        
        self.locationlist = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        if self.device_id is 0:
            self.bariera = barrier.ReusableBarrierCond(len(devices))
            for device in devices:
                device.bariera = self.bariera
                device.locationcondition = self.locationcondition
                device.locationlist = self.locationlist

    def assign_script(self, script, location):
        
        
        
        self.script_lock.acquire()
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()
        self.script_lock.release()

    def get_data(self, location):
        
        
        
        self.data_lock.acquire()
        value = self.sensor_data[location] if location in self.sensor_data\


                                           else None
        self.data_lock.release()
        return value

    def set_data(self, location, data):
        
        
        
        self.data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.data_lock.release()

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
            
            


            self.device.script_lock.acquire()

            
            nodes = []
            for (script, location) in self.device.scripts:
                nodes.append(ScriptThread(self.device, script, location,\
                             neighbours, self.device.locationlist,\
                             self.device.locationcondition))
            for j in xrange(len(self.device.scripts)):
                nodes[j].start()
            for j in xrange(len(self.device.scripts)):
                nodes[j].join()
            
            
            self.device.timepoint_done.clear()
            
            self.device.script_lock.release()
            
            self.device.bariera.wait()

class ScriptThread(Thread):
    

    def __init__(self, device, script, location, neighbours, locationlist,\
                 locationcondition):
        
        Thread.__init__(self, name="Service Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.locationlist = locationlist
        self.locationcondition = locationcondition

    def run(self):
        
        sem = 1
        
        
        
        while sem is 1:
            self.locationcondition.acquire()
            if self.location in self.locationlist:
                self.locationcondition.wait()
            else:
                self.locationlist.append(self.location)
                sem = 0
            self.locationcondition.release()

        
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
        
        
        self.locationcondition.acquire()
        self.locationlist.remove(self.location)
        self.locationcondition.notify_all()
        self.locationcondition.release()
