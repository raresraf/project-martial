




from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    
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
                for _ in range(self.num_threads):
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
        self.thread = DeviceThread(self)
        self.locks = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            locks = []
            barrier = ReusableBarrier(len(devices))

            locations = -1
            for device in devices:
                for location, _ in device.sensor_data.iteritems():
                    if location > locations:
                        locations = location

            for _ in range(0, locations+1):
                locks.append(Lock())
            
            for device in devices:
                device.locks = locks
                device.barrier = barrier
        self.thread.start()

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

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.script_received.wait()
            
            scripts_list = []
            for _ in range(0, 8):
                scripts_list.append([])
            
            index = 0
            for (script, location) in self.device.scripts:
                scripts_list[index].append((script, location))
                index = (index + 1) % 8
            
            list_thread = []
            for lst in scripts_list:
                if len(lst) > 0:
                    list_thread.append(MyThread(self.device, neighbours, lst))
            
            for thr in list_thread:
                thr.start()
            
            for thr in list_thread:
                thr.join()

            
            self.device.barrier.wait()
            
            self.device.script_received.clear()

class MyThread(Thread):
    
    def __init__(self, device, neighbours, lst):
        
        Thread.__init__(self)
        self.device = device
        self.lst = lst


        self.neighbours = neighbours

    def run(self):
        
        for (script, location) in self.lst:
            script_data = []
            
            self.device.locks[location].acquire()
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)
                
                self.device.set_data(location, result)

                
                for device in self.neighbours:
                    device.set_data(location, result)
            self.device.locks[location].release()
        return
