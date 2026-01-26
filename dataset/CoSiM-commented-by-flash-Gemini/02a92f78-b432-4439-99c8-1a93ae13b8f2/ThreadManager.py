


from Queue import Queue
from threading import Thread


class ThreadManager(object):
    
    def __init__(self, threads_count):
        
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None
        self.initialize_workers(threads_count)
    
    def create_workers(self, threads_count):
        
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)
    
    def start_workers(self):
        
        for thread in self.threads:
            thread.start()
    
    def initialize_workers(self, threads_count):
        
        self.create_workers(threads_count)


        self.start_workers()
    
    def set_device(self, device):
        
        self.device = device
    
    def execute(self):
        
        while True:
            neighbours, script, location = self.queue.get()
            no_neighbours = neighbours is None
            no_scripts = script is None
            if no_neighbours and no_scripts:
                self.queue.task_done()
                return
            self.run_script(neighbours, script, location)
            self.queue.task_done()
    
    @staticmethod
    def is_not_empty(given_object):
        
        return given_object is not None
    
    def run_script(self, neighbours, script, location):
        
        script_data = []
        
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if ThreadManager.is_not_empty(data):
                    script_data.append(data)
        
        data = self.device.get_data(location)
        if ThreadManager.is_not_empty(data):
            script_data.append(data)
        if script_data:
            
            result = script.run(script_data)
            
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    continue


                device.set_data(location, result)
            
            self.device.set_data(location, result)
    
    def submit(self, neighbours, script, location):
        
        self.queue.put((neighbours, script, location))
    
    def wait_threads(self):
        
        self.queue.join()
    def end_threads(self):
        
        self.wait_threads()
        
        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)
            
        for thread in self.threads:
            thread.join()


from threading import Condition


class ConditionalBarrier(object):
    
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


from threading import Event, Thread, Lock

from ThreadManager import ThreadManager
from barriers import ConditionalBarrier


class Device(object):
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        self.script_received = Event()
        self.timepoint_done = Event()
        


        self.scripts = []
        self.scripts_arrived = False
        
        self.barrier = None
        self.location_locks = {location: Lock() for location in sensor_data}
        
        self.thread = DeviceThread(self)
        self.thread.start()
    
    def __str__(self):
        
        return "Device %d" % self.device_id
    
    def assign_barrier(self, barrier):
        
        self.barrier = barrier
    
    def setup_devices(self, devices):
        
        number_of_devices = len(devices)
        if self.device_id == 0:


            self.assign_barrier(ConditionalBarrier(number_of_devices))
            self.broadcast_barrier(devices, self.barrier)
    
    @staticmethod
    def broadcast_barrier(devices, barrier):
        
        for device in devices:
            if device.device_id == 0:
                continue
            device.assign_barrier(barrier)
    
    def accept_script(self, script, location):
        
        self.scripts.append((script, location))
        self.scripts_arrived = True
    
    def assign_script(self, script, location):
        
        if script is not None:
            self.accept_script(script, location)
        else:
            self.timepoint_done.set()
    
    def get_data(self, location):
        
        data_is_valid = location in self.sensor_data
        if data_is_valid:
            self.location_locks[location].acquire()
        return self.sensor_data[location] if data_is_valid else None
    
    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
    
    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    
    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadManager(8)
    
    def run(self):
        self.thread_pool.set_device(self.device)
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            while True:
                
                scripts_ready = self.device.scripts_arrived
                done_waiting = self.device.timepoint_done.wait()
                if scripts_ready or done_waiting:
                    if done_waiting and not scripts_ready:
                        self.device.timepoint_done.clear()
                        self.device.scripts_arrived = True
                        break
                    self.device.scripts_arrived = False
                    
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
            
            self.thread_pool.wait_threads()
            
            self.device.barrier.wait()
        
        self.thread_pool.end_threads()
