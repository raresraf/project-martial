


from threading import Event, Thread, Lock, Semaphore


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbours_received = Event()
        self.script_received = Event()
        self.num_scripts = 0
        self.num_threads = 0
        self.scripts = []
        self.threads = []
        self.barrier = None
        self.time_barrier = None
        self.global_lock = None
        self.neighbours = None
        self.location_locks = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if devices[0].device_id == self.device_id:
            time_barrier = ReusableBarrier(len(devices))
            global_lock = Lock()
            for device in devices:
                device.time_barrier = time_barrier
                device.global_lock = global_lock

        self.num_threads = max(min(8, 100 / len(devices)), 1)
        self.barrier = ReusableBarrier(self.num_threads)

        for i in xrange(0, self.num_threads):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(0, self.num_threads):
            self.threads[i].start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.num_scripts = len(self.scripts)
        else:
            self.script_received.set()

    def get_data(self, location):
        
        data = self.sensor_data[location] if location in self.sensor_data else None
        return data

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def acquire_lock(self, location):
        
        my_lock = None
        for (loc, lock) in self.location_locks:
            if location == loc:
                my_lock = lock
        if my_lock is None:
            my_lock = Lock()
            self.location_locks.append((location, my_lock))

        my_lock.acquire()

    def release_lock(self, location):
        
        my_lock = None
        for (loc, lock) in self.location_locks:
            if location == loc:
                my_lock = lock
        if my_lock is None:
            my_lock = Lock()
            self.location_locks.append((location, my_lock))

        my_lock.release()


    def shutdown(self):
        
        for i in xrange(0, self.num_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        while True:
            
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_received.set()



            self.device.neighbours_received.wait()

            neighbours = self.device.neighbours
            if neighbours is None:
                break

            self.device.script_received.wait()

            
            scripts = []
            if self.device.num_scripts <= self.device.num_threads:
                if self.thread_id < self.device.num_scripts:
                    scripts = [self.device.scripts[self.thread_id]]
            else:
                workload_size = self.device.num_scripts / self.device.num_threads
                offset1 = self.thread_id * workload_size
                offset2 = (self.thread_id + 1) * workload_size
                scripts = self.device.scripts[offset1:offset2]
                if self.thread_id == 0:
                    offset1 = self.device.num_threads * workload_size
                    offset2 = self.device.num_scripts
                    scripts += self.device.scripts[offset1:offset2]

            
            peers = []
            for device in neighbours:
                if device != self.device:
                    peers.append(device)
            peers.append(self.device)

            
            for (script, location) in scripts:
                script_data = []

                
                
                self.device.global_lock.acquire()

                for device in peers:
                    device.acquire_lock(location)
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                self.device.global_lock.release()

                
                if script_data != []:
                    result = script.run(script_data)

                
                for device in peers:
                    device.set_data(location, result)
                    device.release_lock(location)

            
            self.device.barrier.wait()
            if self.thread_id == 0:
                self.device.script_received.clear()


                self.device.neighbours_received.clear()
                self.device.time_barrier.wait()
            self.device.barrier.wait()



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
