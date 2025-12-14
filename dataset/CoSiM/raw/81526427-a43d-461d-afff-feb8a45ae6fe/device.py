


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    
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
                i = 0
                while i < self.num_threads:
                    threads_sem.release()
                    i = i + 1
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.barrier2 = None
        self.max_threads = 8
        self.thread = DeviceThread(self)
        self.barrier_lock = Lock()
        self.location_locks = {}
        for loc in sensor_data.keys():
            self.location_locks[loc] = None
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        index = 0
        
        
        self.barrier_lock = devices[index].barrier_lock

        self.barrier_lock.acquire()
        
        barrier_list = [d.barrier for d in devices if d.barrier is not None]
        barrier_list2 = [d.barrier2 for d in devices if d.barrier2 is not None]
        loc_list = [device.location_locks for device in devices]

        
        
        for loc in loc_list:
            for val in loc.keys():
                if val not in self.location_locks.keys():
                    self.location_locks[val] = loc[val]
                elif loc[val] is not None and self.location_locks[val] is None:
                    self.location_locks[val] = loc[val]

        
        keys = self.location_locks.keys()
        rest = [index for index in keys if self.location_locks[index] is None]
        for index in rest:
            self.location_locks[index] = Lock()

        
        index = 0
        if len(barrier_list) == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            self.barrier2 = ReusableBarrierSem(len(devices))
        else:
            self.barrier = barrier_list[index]
            self.barrier2 = barrier_list2[index]

        self.barrier_lock.release()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.barrier.wait()
            self.timepoint_done.set()

    def get_data(self, location):
        
        result = None
        if location in self.sensor_data:
            result = self.sensor_data[location]
        return result

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.num_threads = 1

    def run_scripts(self, script, location, neighbours):
        
        script_data = []

        
        
        self.device.location_locks[location].acquire()

        
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        if script_data != []:
            
            result = script.run(script_data)
            
            for device in neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)

        
        self.device.location_locks[location].release()

    def run(self):
        
        while True:
            


            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                self.device.barrier.wait()
                break
            self.device.barrier.wait()
            
            child_threads = []
            
            for (script, location) in self.device.scripts:
                if self.num_threads < self.device.max_threads:
                    
                    self.num_threads = self.num_threads + 1
                    arguments = (script, location, neighbours)
                    child = Thread(target=self.run_scripts, args=arguments)
                    child_threads.append(child)
                    child.start()
                else:
                    self.run_scripts(script, location, neighbours)
            for child in child_threads:
                
                child.join()
                self.num_threads = self.num_threads - 1
            
            self.device.barrier2.wait()
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
