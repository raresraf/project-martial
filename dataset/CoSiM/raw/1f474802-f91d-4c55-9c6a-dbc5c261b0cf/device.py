


from threading import Event, Thread, Lock
from barrier import Barrier
from thread_pool import ThreadPool

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
        self.barrier = None
        self.locks = {location : Lock() for location in sensor_data}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(8)

    def run(self):
        


        self.thread_pool.device = self.device
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            while True:
                
                if self.device.script_received.isSet():
                    self.device.script_received.clear()
                    
                    for (script, location) in self.device.scripts:
                        self.thread_pool.queue.put((neighbours, script, location))
                
                elif self.device.timepoint_done.wait():
                    
                    if self.device.script_received.isSet():


                        self.device.script_received.clear()
                        
                        for (script, location) in self.device.scripts:
                            self.thread_pool.queue.put((neighbours, script, location))
                    
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break
            
            self.thread_pool.queue.join()
            
            self.device.barrier.wait()
        
        
        self.thread_pool.queue.join()
        
        for _ in xrange(len(self.thread_pool.threads)):
            self.thread_pool.queue.put((None, None, None))
        
        for thread in self.thread_pool.threads:
            thread.join()


from Queue import Queue
from threading import Thread

class ThreadPool(object):
    

    def __init__(self, num_threads):
        
        self.queue = Queue(num_threads)
        self.threads = []
        self.device = None
        self.workers(num_threads)
        self.start_workers()

    def workers(self, num_threads):
        
        for _ in xrange(num_threads):
            new_thread = Thread(target=self.run)
            self.threads.append(new_thread)

    def start_workers(self):
        
        for thread in self.threads:
            thread.start()

    def run(self):
        


        while True:
            neighbours, script, location = self.queue.get()
            
            if neighbours is None and script is None:
                self.queue.task_done()
                return
            script_data = []
            
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
            if script_data != []:
                
                result = script.run(script_data)
                
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                self.device.set_data(location, result)
            self.queue.task_done()
