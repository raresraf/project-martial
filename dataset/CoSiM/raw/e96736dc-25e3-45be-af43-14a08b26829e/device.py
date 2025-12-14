


from threading import Event, Thread, Condition, Lock, Semaphore
from random import shuffle


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.setup_ev = Event()
        self.barrier = None
        self.lock = None
        self.get_lock = None
        self.scripts = []
        self.script_locations = {}  
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        

        self.devices = devices

        if (self.device_id == 0):   
            self.barrier = ReusableBarrier(len(devices))    
            self.lock = Lock()  
            self.get_lock = Lock()

            for i in xrange(1, len(devices)):
                devices[i].barrier = self.barrier
                devices[i].lock = self.lock
                devices[i].get_lock = self.get_lock
                devices[i].script_locations = self.script_locations
                devices[i].setup_ev.set()   

            self.setup_ev.set()

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
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
        self.no_cores = 8
        self.semaphore = Semaphore(self.no_cores + 1)    

    def run(self):
        
        self.device.setup_ev.wait()


        while True:
            
            self.device.get_lock.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.get_lock.release()

            if neighbours is None:
                break

            self.device.script_received.wait()
            self.device.script_received.clear()

            calcs = []  

            for i in xrange(0, len(self.device.scripts)):
                with self.device.lock:  
                    if not self.device.script_locations.has_key(self.device.scripts[i][1]):
                        
                        self.device.script_locations[self.device.scripts[i][1]] = Lock()
                calcs.append(ScriptCalculator(i, self.device, neighbours, self.device.scripts[i], self.semaphore))
                self.semaphore.acquire()
                calcs[i].start()
                self.semaphore.release()

            for i in xrange(0, len(self.device.scripts)):   
                calcs[i].join()

            self.device.barrier.wait()  


class ReusableBarrier():    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
        self.lock = Lock()

    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()


        self.cond.release()


class ScriptCalculator(Thread):

    def __init__(self, name, device, neighbours, script, semaphore):
        Thread.__init__(self)
        self.name = name
        self.device = device
        self.neighbours = neighbours
        self.scripts = script
        self.semaphore = semaphore

    def run(self):
        self.semaphore.acquire()    

        script = self.scripts[0]
        location = self.scripts[1]

        self.device.script_locations[location].acquire()    
        script_data = []    

        
        for device in self.neighbours:
            data = device.get_data(location)
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
        self.device.script_locations[location].release()    
        self.semaphore.release()    