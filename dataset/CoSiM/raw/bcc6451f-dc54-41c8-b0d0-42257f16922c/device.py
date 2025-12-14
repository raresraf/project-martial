


from threading import Event, Thread, Lock, Condition

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
        self.barrier = ReusableBarrierCond(0)
        self.dict = {}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            idroot = 0
            
            self.barrier = ReusableBarrierCond(len(devices))
            for j in xrange(len(devices)):
                
                if devices[j].device_id == 0:
                    idroot = j
                
                for location in devices[j].sensor_data:
                    self.dict[location] = Lock()
            for k in xrange(len(devices)):
                
                devices[k].barrier = devices[idroot].barrier
                
                for j in xrange(len(self.dict)):
                    devices[k].dict[j] = self.dict[j]

        

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
            
            self.device.barrier.wait()
            if neighbours is None:
                break
            self.device.timepoint_done.wait()
            if self.device.scripts is None:
                break
            threadsnew = []
            
            for j in xrange(8):
                lis = []
                k = 0
                for (script, loc) in self.device.scripts:
                    if k % 8 == j:
                        lis.append((script, loc))
                    k = k + 1


                threadsnew.append(MyThread(self.device, neighbours, lis))
            for thread in threadsnew:
                thread.start()
            for thread in threadsnew:
                thread.join()
            
            self.device.timepoint_done.clear()

class MyThread(Thread):
    
    def __init__(self, device, neighbours, lis):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.lis = lis

    def run(self):
        
        for (script, location) in self.lis:
            self.device.dict[location].acquire()
            script_data = []
            
            for device in self.neighbours:
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
            if script_data != []:
                
                result = script.run(script_data)
                
                
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)
            self.device.dict[location].release()

class ReusableBarrierCond():
    
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
        
