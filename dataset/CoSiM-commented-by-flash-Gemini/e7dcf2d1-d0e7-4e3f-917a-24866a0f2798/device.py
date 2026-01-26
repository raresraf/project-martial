


from threading import Event, Thread, Lock, Semaphore


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

        
        self.global_barrier = None
        
        self.locks = None


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
        	
            self.global_barrier = ReusableBarrier(len(devices))

            for device in devices:
                device.global_barrier = self.global_barrier

            
            self.locks = []
            locations = devices[0].sensor_data.keys()
            for index in range(1, len(devices)):
                aux = devices[index].sensor_data.keys()
                locations = list(set(locations).union(aux))

            
            for _ in range(len(locations)):
                self.locks.append(Lock())

            for device in devices:
                device.locks = self.locks


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()

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

        
        self.barrier_parent = ReusableBarrier(9)

        
        self.threads = []
        for _ in range(8):
            self.threads.append(Worker(self.device, None, None, self.barrier_parent))

        for thread in self.threads:
            thread.start()


    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()
            self.device.script_received.clear()

            
            if len(self.device.scripts) <= 8:
                for index in range(len(self.device.scripts)):
                    self.threads[index].script = self.device.scripts[index]
                    self.threads[index].neighbours = neighbours
            else:
                
                aux = len(self.device.scripts)/8
                inf = 0
                sup = aux
                for index in range(8):
                    if index == 7:
                        sup = len(self.device.scripts)
                    self.threads[index].neighbours = neighbours
                    for index2 in range(inf, sup):
                        self.threads[index].script = self.device.scripts[index2]
                    inf += aux
                    sup += aux


            
            self.barrier_parent.wait()

            self.device.timepoint_done.wait()


            self.device.timepoint_done.clear()

            
            self.barrier_parent.wait()
            
            self.device.global_barrier.wait()

        
        for thread in self.threads:
            thread.out = 1
        self.barrier_parent.wait()

        
        for thread in self.threads:
            thread.join()


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


class Worker(Thread):
    

    def __init__(self, device, script, neighbours, barrier_parent):
        
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device

        self.script = script
        self.neighbours = neighbours
        self.workers_barrier = barrier_parent
        self.out = 0

    def run(self):
        while True:
            self.workers_barrier.wait()

            if self.out == 1:
                break

            if self.neighbours != None:
                
                script_data = []

                
                self.device.locks[self.script[1]].acquire()
                


                for device in self.neighbours:
                    data = device.get_data(self.script[1])
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(self.script[1])
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = self.script[0].run(script_data)

                    
                    for device in self.neighbours:
                        device.set_data(self.script[1], result)
                    
                    self.device.set_data(self.script[1], result)
                
                self.device.locks[self.script[1]].release()
            
            self.workers_barrier.wait()
