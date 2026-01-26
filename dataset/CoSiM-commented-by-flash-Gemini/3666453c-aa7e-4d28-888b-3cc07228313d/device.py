


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
                i = 0
                while i < self.num_threads:
                    threads_sem.release()
                    i += 1
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.max_threads = 8
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.notification = Event()
        self.timepoint_done = Event()
        self.notification.clear()
        self.timepoint_done.clear()
        self.update_locks = {}
        self.read_locations = {}
        self.external_barrier = None
        self.internal_barrier = ReusableBarrier(self.max_threads)
        self.workers = self.setup_workers()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_workers(self):
        
        workers = []
        i = 0
        while i < self.max_threads:
            workers.append(Worker(self))
            i += 1
        return workers

    def start_workers(self):
        
        for i in range(0, self.max_threads):
            self.workers[i].start()

    def stop_workers(self):
        
        for i in range(0, self.max_threads):
            self.workers[i].join()

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.external_barrier = ReusableBarrier(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    while device.external_barrier is None:
                        pass
                    self.external_barrier = device.external_barrier
                    break

    def assign_script(self, script, location):
        
        self.notification.set()
        if script is not None:
            if location not in self.update_locks:
                self.update_locks[location] = Lock()
                self.read_locations[location] = Event()
                self.read_locations[location].set()
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location not in self.sensor_data:
            return None
        else:
            if location not in self.read_locations:
                self.update_locks[location] = Lock()
                self.read_locations[location] = Event()
                self.read_locations[location].set()
            self.read_locations[location].wait()
            return self.sensor_data[location]

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.update_locks[location].acquire()
            self.read_locations[location].clear()
            self.sensor_data[location] = data
            self.read_locations[location].set()
            self.update_locks[location].release()

    def shutdown(self):
        
        self.stop_workers()
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def find_free_worker(self):
        
        for i in range(0, self.device.max_threads):
            if self.device.workers[i].is_free:
                return i
        return -1

    def run(self):
        
        self.device.start_workers()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            
            if neighbours is None:
                for i in range(0, self.device.max_threads):
                    self.device.workers[i].update(None, None, None, "end")
                break

            
            if len(self.device.scripts) == 0:
                self.device.notification.wait()

            
            curr_scr = 0
            while (curr_scr < len(self.device.scripts)) or \
                  (self.device.timepoint_done.is_set() is False):
                worker_idx = self.find_free_worker()
                if (worker_idx >= 0) and (curr_scr < len(self.device.scripts)):
                    (script, location) = self.device.scripts[curr_scr]
                    self.device.workers[worker_idx].update(location, script, neighbours, "run")
                    curr_scr += 1
                else:
                    continue

            
            for i in range(0, self.device.max_threads):
                self.device.workers[i].update(None, None, None, "timepoint_end")
            self.device.timepoint_done.clear()
            self.device.notification.clear()
            
            self.device.external_barrier.wait()


class Worker(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self)
        self.device = device
        self.init_start = Event()
        self.exec_start = Event()
        self.location = None
        self.script = None
        self.neighbours = None
        self.is_free = True
        self.mode = ""
        self.exec_start.clear()
        self.init_start.set()

    def update(self, location, script, neighbours, mode):
        
        self.init_start.wait()
        self.init_start.clear()
        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.mode = mode
        self.is_free = False
        self.exec_start.set()

    def run(self):
        
        while True:
            self.exec_start.wait()
            self.exec_start.clear()
            if self.mode == "end":
                
                break
            elif self.mode == "timepoint_end":
                
                self.device.internal_barrier.wait()
                self.is_free = True
                self.init_start.set()
            else:
            	
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
                self.is_free = True
                self.init_start.set()
