


from threading import Thread, Semaphore, Event, Lock

class ReusableBarrierSem(object):
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)
    def wait(self):
        
        self.phase1()
        self.phase2()
    def phase1(self):
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):


                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    def phase2(self):
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

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
        self.locks = []
        
        self.nrlocks = max(sensor_data)
    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            
            for _, device in enumerate(devices):
                device.barrier = self.barrier
        
        if self.device_id == 0:
            listmaxim = []
            for _, device in enumerate(devices):
                listmaxim.append(device.nrlocks)
            
            number = max(listmaxim)
            
            for _ in range(number + 1):
                self.locks.append(Lock())
            
            for _, device in enumerate(devices):
                device.locks = self.locks
    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        

        if location in self.sensor_data:
            data = self.sensor_data[location]
        else: data = None
        return data
    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
    def shutdown(self):
        
        self.thread.join()

class MiniDeviceThread(Thread):
    
    def __init__(self, device, script, location, neighbours):
        
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
	
        self.device.locks[self.location].acquire()
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
        
        self.device.locks[self.location].release()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.nr_iter = None

    def run(self):

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.timepoint_done.wait()
            
            self.nr_iter = len(self.device.scripts) / 8
            
            if self.nr_iter == 0:
                scriptthreads = []
                for (script, location) in self.device.scripts:
                    scriptthreads.append(MiniDeviceThread(self.device, script, location, neighbours))
                
                for _, thread in enumerate(scriptthreads):
                    thread.start()
                
                for _, thread in enumerate(scriptthreads):
                    thread.join()
            
            
            else:
                count = 0
                size = 8
                for _ in range(self.nr_iter):
                    scriptthreads = []
                    for idx in range(count, size):
                        script = self.device.scripts[idx][0]
                        location = self.device.scripts[idx][1]
                        scriptthreads.append(MiniDeviceThread(self.device, script, location, neighbours))
                    
                    for _, thread in enumerate(scriptthreads):
                        thread.start()
	                
                    for _, thread in enumerate(scriptthreads):
                        thread.join()
                    count = count + 8
                    if size + 8 > len(self.device.scripts):
                        size = len(self.device.scripts) - size
                    else:
                        size = size + 8
            
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
