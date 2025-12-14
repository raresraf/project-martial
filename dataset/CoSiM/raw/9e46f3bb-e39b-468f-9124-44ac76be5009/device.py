


from threading import Event, Thread, Lock
from Queue import Queue
from barrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_done = Event()
        self.scripts_already_parsed = Event()
        self.queue = Queue()
        self.barrier = None
        self.location_locks = None
        self.neighbours = None
        self.thread = DeviceThread(self)
        self.thread.start()
        self.workers = []
        for _ in xrange(8):
            worker = Worker(self)
            worker.start()
            self.workers.append(worker)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        
        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            self.location_locks = {}

            for device in devices:
                device.location_locks = self.location_locks
                device.barrier = self.barrier
                device.setup_done.set()
        else:
            self.setup_done.wait()

    def assign_script(self, script, location):
        
        if script is not None:
            if location not in self.location_locks:     
                self.location_locks[location] = Lock()  
            self.scripts.append((script, location))
            if self.scripts_already_parsed.is_set():    
                self.queue.put((script, location))      
                                                        
                                                        
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()
        for i in xrange(8):
            self.queue.put((None, None))    
                                            
        for i in xrange(8):
            self.workers[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            self.device.scripts_already_parsed.clear()      
                                                            
                                                            
            


            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                break

            
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location))
            self.device.scripts_already_parsed.set()        
                                                            
                                                            
                                                            
                                                            

            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            self.device.queue.join()
            self.device.barrier.wait()


class Worker(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Worker")
        self.device = device

    def run(self):
        while True:
            (script, location) = self.device.queue.get()
            if script is None:
                break



            with self.device.location_locks[location]:  
                                                        
                                                        
                                                        
                                                        
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

            self.device.queue.task_done()
