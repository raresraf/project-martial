


from threading import Lock, Thread, Event, Semaphore
import math
from fractions import Fraction


class ReusableBarrier(object):
    
    def __init__(self, threads):
        self.threads = threads
        self.count_threads1 = [self.threads]
        self.count_threads2 = [self.threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.threads):
                    threads_sem.release()
                    
                    i = i
                count_threads[0] = self.threads
        threads_sem.acquire()


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.barrier = None
        self.locations_lock = []
        self.lock = Lock()
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
        
        locations = 50

        

        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            for i in range(locations):


                self.locations_lock.append(Lock())
                i = i
            for device in devices:
                device.barrier = barrier
                device.locations_lock = self.locations_lock

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


class Slave(Thread):
    
    def __init__(self, neighbours, device):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.device = device
        self.scripts = []

    
    def give_work(self, work):
        
        self.scripts.append(work)

    def run(self):
        
        


        for (script, location) in self.scripts:
            
            with self.device.locations_lock[location]:
                script_data = []
                
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    


                    result = script.run(script_data)

                    
                    for device in self.neighbours:
                        device.set_data(location, result)
                    

                    self.device.set_data(location, result)


class DeviceThread(Thread):
    
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.slave_pool = []

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.timepoint_done.wait()
            
            num_threads = 8
            
            self.slave_pool = []
            
            for i in range(min(num_threads, len(self.device.scripts))):
                helper = Slave(neighbours, self.device)
                self.slave_pool.append(helper)
            
            equally_work = 0
            i = 0
            
            if len(self.slave_pool):
                for (script, location) in self.device.scripts:
                    self.slave_pool[i].give_work((script, location))
                    equally_work += 1
                    
                    if equally_work == math.ceil(Fraction(len(self.device.scripts), num_threads)):
                        i += 1
                        equally_work = 0
                    if i == len(self.slave_pool):
                        i = 0
                        equally_work = 0
                
                for slv in range(len(self.slave_pool)):
                    self.slave_pool[slv].start()
                
                for cls in range(len(self.slave_pool)):
                    self.slave_pool[cls].join()

            self.device.timepoint_done.clear()
            
            self.device.barrier.wait()


