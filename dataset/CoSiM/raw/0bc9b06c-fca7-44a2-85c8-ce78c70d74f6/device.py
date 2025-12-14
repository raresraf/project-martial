



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
        self.scripts_received = Event()
        self.all_devices = []
        self.scripts = []
        self.data_lock = Lock()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = ReusableBarrier(0)
        self.location_locks = {}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            
            for location in xrange(100):
                self.location_locks[location] = Lock()
            self.barrier = ReusableBarrier(len(devices))

        for dev in devices:
            
            
            if self.device_id == 0:
                dev.barrier = self.barrier
                dev.location_locks = self.location_locks
            self.all_devices.append(dev)

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            
            self.scripts_received.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        
        
        self.data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.data_lock.release()

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.assigned_scripts = {}

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.scripts_received.wait()


            self.device.scripts_received.clear()

            
            device_threads = []
            for (script, location) in self.device.scripts:
                device_threads.append(
                    DeviceSubThread(self.device, script, location, neighbours)
                )

            
            for i in xrange(len(device_threads)):
                device_threads[i].start()

            
            for i in xrange(len(device_threads)):
                device_threads[i].join()

            
            self.device.barrier.wait()

class DeviceSubThread(Thread):
    

    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        
        self.device.location_locks[self.location].acquire()
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

        
        self.device.location_locks[self.location].release()
