

from threading import Event, Thread, Condition, Lock
from Queue import Queue

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []       
        self.locks = {}         
                                    
        self.no_more_scripts = Event()  
                                            
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))

        
        for device in devices:
            if device is not self:
                device.set_barrier(self.barrier)


    def assign_script(self, script, location):
        
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            
            self.no_more_scripts.set()

    def get_data(self, location):
        
        


        if location in self.sensor_data:
            
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            
            return None

    def set_data(self, location, data):
        
        
        if location in self.sensor_data:
            
            self.sensor_data[location] = data
            self.locks[location].release()

    
    def set_barrier(self, barrier):
        
        self.barrier = barrier

    def shutdown(self):
        
        

        for thread in self.thread.child_threads:
            if thread.is_alive():
                thread.join()
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue()        
        self.child_threads = []     
        self.max_threads = 8        


    def run(self):

        
        for location, data in self.device.sensor_data.iteritems():
            self.device.locks[location] = Lock()

        
        for i in xrange(self.max_threads):
            thread = Thread(target=process_scripts, args=(self.queue,))
            self.child_threads.append(thread)
            thread.start()

        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            
            if neighbours is None:
                
                for i in xrange(len(self.child_threads)):
                    self.queue.put(None)
                
                self.queue.join()
                break

            done_scripts = 0
            
            for (script, location) in self.device.scripts:
                
                job = {}
                job['script'] = script
                job['location'] = location
                job['device'] = self.device
                job['neighbours'] = neighbours
                self.queue.put(job)     
                done_scripts += 1       

            
            self.device.no_more_scripts.wait()
            
            self.device.no_more_scripts.clear()
            
            if done_scripts < len(self.device.scripts):
                for (script, location) in self.device.scripts[done_scripts:]:
                    
                    job = {}
                    job['script'] = script
                    job['location'] = location
                    job['device'] = self.device
                    job['neighbours'] = neighbours
                    self.queue.put(job)     

            
            self.queue.join()

            
            self.device.barrier.wait()

def process_scripts(queue):
    
    while True:
        
        job = queue.get()
        
        if job is None:
            queue.task_done()
            break
        
        script = job['script']
        location = job['location']
        mydevice = job['device']
        neighbours = job['neighbours']

        script_data = []
        
        for device in neighbours:
            if device is not mydevice:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        
        data = mydevice.get_data(location)


        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data)

            
            for device in neighbours:
                if device is not mydevice:
                    device.set_data(location, result)
            
            mydevice.set_data(location, result)
        
        queue.task_done()



class ReusableBarrier(object):
    

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
