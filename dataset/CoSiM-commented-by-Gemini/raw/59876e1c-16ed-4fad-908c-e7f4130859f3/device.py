




from threading import Thread, Lock
import barrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.barrier = None
        self.locationslocks = {}
        self.neighbours = []
        self.workpool = Workpool()
        self.thread = DeviceThread(self)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            
            locationslocks = {}
            
            for dev in devices:
                for location in dev.sensor_data:
                    locationslocks[location] = Lock()

            
            barr = barrier.ReusableBarrierCond(len(devices))

            for dev in devices:
                dev.locationslocks = locationslocks
                dev.barrier = barr
                dev.thread.start()



    def assign_script(self, script, location):
        
        if script is not None:
            
            self.scripts.append((script, location))

            
            self.workpool.putwork(script, location)
        else:
            
            self.workpool.endwork()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []
        
        self.workerbar = barrier.ReusableBarrierCond(9)

    def run(self):
        while True:
            
            self.device.workpool.putlistwork(self.device.scripts)

            
            self.device.neighbours = self.device.supervisor.get_neighbours()
            neighbours = self.device.neighbours

            
            if neighbours is None:
                break

            
            for i in xrange(8):
                worker = Worker(self.workerbar, self.device)
                self.workers.append(worker)
                self.workers[i].start()

            
            
            self.workerbar.wait()

            
            for i in range(8):
                self.workers[i].join()
            del self.workers[:]

            
            self.device.barrier.wait()


class Workpool(object):
    

    def __init__(self):
        
        self.scripts = [] 
        self.lock = Lock()
        self.done = False


    def getwork(self):
        
        self.lock.acquire()
        
        if self.done is False or len(self.scripts) > 0:
            
            if len(self.scripts) > 0:
                pair = self.scripts.pop()
                self.lock.release()
                return pair
            else:
                self.lock.release()
                return ()
        else:
            
            self.lock.release()
            return None


    def putwork(self, script, location):
        
        self.lock.acquire()
        self.scripts.append((script, location))
        self.lock.release()

    def endwork(self):
        
        self.lock.acquire()
        self.done = True
        self.lock.release()

    def putlistwork(self, scripts):
        
        self.lock.acquire()
        self.done = False
        self.scripts = list(scripts)
        self.lock.release()


class Worker(Thread):
    

    def __init__(self, barr, device):
        
        Thread.__init__(self, name="Worker Thread")
        self.lock = Lock()
        self.barrier = barr
        self.device = device

    def run(self):
        
        while True:
            
            work = self.device.workpool.getwork()

            if work is None:
                
                self.barrier.wait()
                return
            else:
                
                if work is not ():
                    self.update(work[0], work[1])

    def update(self, script, location):
        
        


        self.device.locationslocks[location].acquire()

        script_data = []
        
        for device in self.device.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            


            result = script.run(script_data)

            
            for device in self.device.neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)

        self.device.locationslocks[location].release()
