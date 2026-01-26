


from threading import Event, Thread, Semaphore, Lock, RLock
from reusable_barrier import ReusableBarrier
import multiprocessing

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.results = {}
        self.lock = None 
        self.dislocksdict = None 
        self.barrier = None
        self.sem = Semaphore(1)
        self.sem2 = Semaphore(0)
        self.all_devices = []
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
        
        loc = []
        for d in devices:
            for l in d.sensor_data:
                loc.append(l) 
        all_devices = devices
        if self.device_id == 0:
            self.sem2.release()
            self.barrier = ReusableBarrier(len(devices))
            self.dislocksdict = {}
            for k in list(set(loc)):
                self.dislocksdict[k] = RLock()
            self.lock = Lock()

        self.sem2.acquire()

        for d in devices:
            if d.barrier == None:
                d.barrier = self.barrier 
                d.sem2.release() 
                d.dislocksdict = self.dislocksdict
                d.lock = Lock()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
   
    def get_data(self, location):
        
        data = -1
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None

    def set_data(self, location, data):
        
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class RunScript(Thread):
    def __init__(self, script, location, neighbours, device):
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device
    def run(self):
        
        self.device.dislocksdict[self.location].acquire()
        script_data = []
        for device in self.neighbours:  
            device.lock.acquire()
            data = device.get_data(self.location) 
            device.lock.release()


            if data is not None:
                script_data.append(data)
                
        self.device.lock.acquire()
        data = self.device.get_data(self.location)
        self.device.lock.release()
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = self.script.run(script_data) 
            
            for device in self.neighbours:
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()
        self.device.dislocksdict[self.location].release()


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
            self.device.barrier.wait() 
            script_threads = []
            for (script, location) in self.device.scripts:
                script_threads.append(RunScript(script, location, neighbours, self.device))
            for t in script_threads:
                t.start() 
            for t in script_threads:
                t.join() 
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
