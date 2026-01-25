


from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.loopBarrier = None
        self.locationSemaphores = None
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        loopBarrier = ReusableBarrierCond(len(devices))
        locationSemaphores = {}
        for device in devices :
            device.loopBarrier = loopBarrier
            device.locationSemaphores = locationSemaphores

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            if self.locationSemaphores.get(location) is None:
                self.locationSemaphores[location] = Semaphore()
        else:
            
            self.script_received.set()

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
        self.workLock = Lock()
        self.lastScriptGiven = 0;

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.lastScriptGiven = 0;

            
            self.device.script_received.wait()
            
            
            workers = []
            workLock = Lock()


            for i in range(0, min(8, len(self.device. scripts))):
                worker = ScriptThread(self, neighbours, workLock)
                workers.append(worker)

            
            for worker in workers:
                worker.start()

            
            for worker in workers:
                worker.join()

            
            self.device.script_received.clear()

            
            self.device.loopBarrier.wait()

    def getWork(self):
        

        script = None
        if (self.lastScriptGiven < len(self.device.scripts)):
            script = self.device.scripts[self.lastScriptGiven]
            self.lastScriptGiven += 1

        return script

class ScriptThread(Thread) :
    
    
    def __init__(self, master, neighbours, workLock):
        
        Thread.__init__(self)
        self.master = master
        self.neighbours = neighbours
        self.workLock = workLock

    def run(self) :
        
        self.workLock.acquire()
        scriptLocation = self.master.getWork()
        self.workLock.release()

        
        while scriptLocation is not None:
            (script, location) = scriptLocation
            script_data = []
            
            
            self.master.device.locationSemaphores.get(location).acquire()
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.master.device.get_data(location)
            if data is not None:
                script_data.append(data)



            if script_data != []:
                
                result = script.run(script_data)

                
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.master.device.set_data(location, result)

            
            self.master.device.locationSemaphores.get(location).release()
            
            self.workLock.acquire()
            scriptLocation = self.master.getWork()
            self.workLock.release()