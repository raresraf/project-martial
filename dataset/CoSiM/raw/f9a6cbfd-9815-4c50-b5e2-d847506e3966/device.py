



from threading import Event, Thread, Semaphore, Lock

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
        

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.lock = Lock()
        self.locs = []
        self.hashset = {}
        self.bariera = ReusableBarrier(1)
        
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            self.hashset = {}
            for device in devices:
                for location in device.sensor_data:
                    self.hashset[location] = Lock()
            self.bariera = ReusableBarrier(len(devices))
            for device in devices:
                device.bariera = self.bariera
                device.hashset = self.hashset

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        
        self.lock.acquire()
        aux = self.sensor_data[location] if location in self.sensor_data else None
        self.lock.release()
        return aux

    def set_data(self, location, data):
        
        
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        
        self.thread.join()


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
            
            list_threads = []
            for (script, location) in self.device.scripts:
                list_threads.append(ScriptThread(self.device, script,
                location, neighbours))
            
            for i in xrange(len(list_threads)):
                list_threads[i].start()
            
            for i in xrange(len(list_threads)):
                list_threads[i].join()
            self.device.timepoint_done.clear()
            self.device.bariera.wait()

class ScriptThread(Thread):
    

    def __init__(self, device, script, location, neighbours):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        
        self.device.hashset[self.location].acquire()
        script_data = []
        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        self.device.hashset[self.location].release()
