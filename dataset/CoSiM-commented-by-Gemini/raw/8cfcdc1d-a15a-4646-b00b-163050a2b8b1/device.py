


from threading import Event, Thread, Lock, Semaphore
from collections import deque


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        
        self.thread = []
        
        self.local_lock = Lock()
        
        self.zones = None
        self.num_threads = 8
        self.got_neighbours = Event()


        self.got_scripts = Event()
        self.neighbours = []
        
        self.zones_lock = None
        
        self.local_barrier = ReusableBarrier(self.num_threads)
        
        self.global_barrier = ReusableBarrier(1)
        self.todo_scripts = None

        for _ in range(self.num_threads):
            self.thread.append(DeviceThread(self))


        for i in range(self.num_threads):
            self.thread[i].start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        zones = {}
        
        global_barrier = ReusableBarrier(devices[0].num_threads * len(devices))
        
        zones_lock = Lock()
        for dev in devices:
            dev.zones = zones
            dev.global_barrier = global_barrier
            dev.zones_lock = zones_lock

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
        
        for thread in self.thread:
            thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        self.device.got_neighbours.set()
        self.device.got_scripts.set()
        while True:
            
            self.device.global_barrier.wait()

            
            self.device.local_lock.acquire()
            if self.device.got_neighbours.isSet():
                self.device.timepoint_done.clear()
                
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.got_neighbours.clear()

            
            self.device.local_lock.release()


            self.device.local_barrier.wait()
            self.device.got_neighbours.set()
            if self.device.neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            
            self.device.local_lock.acquire()
            if self.device.got_scripts.isSet():
                self.device.todo_scripts = deque(self.device.scripts)
                self.device.got_scripts.clear()
            self.device.local_lock.release()
            self.device.local_barrier.wait()
            self.device.got_scripts.set()

            while True:
                self.device.local_lock.acquire()
                
                if not self.device.todo_scripts:
                    
                    self.device.local_lock.release()
                    break

                
                (script, location) = self.device.todo_scripts.popleft()

                self.device.zones_lock.acquire()
                if location not in self.device.zones.keys():
                    self.device.zones[location] = Lock()
                self.device.zones_lock.release()

                
                self.device.zones[location].acquire()
                self.device.local_lock.release()

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
                self.device.zones[location].release()


class ReusableBarrier():
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
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
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
                                                 
