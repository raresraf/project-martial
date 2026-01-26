


from threading import Event, Thread, Semaphore, Lock

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
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class ScriptThread(Thread):
    
    def __init__(self, script, location, device, neighbours):
        
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        
        self.device.hash_locatie[self.location].acquire()

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

            self.device.lock.acquire()
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
            self.device.lock.release()

        self.device.hash_locatie[self.location].release()

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barrier = None
        self.lock = None
        self.hash_locatie = None
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == devices[0].device_id:
            barrier = ReusableBarrier(len(devices))
            my_lock = Lock()
            hash_locatie = {}
            for i in range(101):
                hash_locatie[i] = Lock()
            self.barrier = barrier
            self.lock = my_lock
            self.hash_locatie = hash_locatie
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
                if device.lock is None:
                    device.lock = my_lock
                if device.hash_locatie is None:
                    device.hash_locatie = hash_locatie

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

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



            self.device.timepoint_done.wait()

            script_list = []

            
            for (script, location) in self.device.scripts:
                script_list.append(ScriptThread(script,
                                                location,
                                                self.device,
                                                neighbours))

            
            for thread in script_list:
                thread.start()

            
            for thread in script_list:
                thread.join()

            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
