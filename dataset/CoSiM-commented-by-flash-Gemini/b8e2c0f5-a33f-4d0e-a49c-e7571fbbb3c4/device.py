




from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty


class ReusableBarrierCond():
    
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()                  
                                                 
    
    def wait(self):
        self.cond.acquire()                      
        self.count_threads -= 1;
        if self.count_threads == 0:
            self.cond.notify_all()               
            self.count_threads = self.num_threads
        else:
            self.cond.wait();                    
        self.cond.release();                     


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.myThread = myThread(self)
        self.myThread.start()
        
        self.neightbours = []
        self.barrier = None
        self.locks = []
        self.queue = Queue()
        self.programEnded = False;
        
        self.deviceThreads = []
        for i in xrange(8):
            worker = DeviceThread(self)
            self.deviceThreads.append(worker)
            worker.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        devicesNumber = len(devices)
        if self.device_id == 0:


            barrier = ReusableBarrierCond(devicesNumber)
            locks = [Lock() for _ in xrange(24)]
            
            
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.myThread.join()
        for deviceThread in self.deviceThreads:
            deviceThread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        while not self.device.programEnded:

            self.device.script_received.wait()

            try:
                
                script, location = self.device.queue.get(block = False)
                
                
                self.device.locks[location].acquire()
            
                script_data = []
                
                
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:


                        script_data.append(data)
                        
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if len(script_data) > 0:
                    
                    result = script.run(script_data)

                    
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)


                self.device.locks[location].release()
                
                self.device.queue.task_done()
            
            except Empty:
                pass
            
class myThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device myThread %d" % device.device_id)


        self.device = device

    def run(self):
    
        while True:
            
            
            self.device.neighbours = self.device.supervisor.get_neighbours()
            
            
            if self.device.neighbours is None:
                self.device.programEnded = True;
                self.device.script_received.set()
                break;
            
            
            self.device.timepoint_done.wait()
            
            
            for script in self.device.scripts:
                self.device.queue.put(script)
                
            
            self.device.script_received.set()
            
            
            self.device.queue.join()
            
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            
            self.device.barrier.wait()