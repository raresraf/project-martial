


from threading import Event, Thread, Lock, Semaphore

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
                i = self.num_threads
                while i > 0:
                    self.threads_sem1.release()
                    i -= 1


                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                i = self.num_threads
                while i > 0:
                    self.threads_sem2.release()
                    i -= 1
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.script_received = Event()
        self.locations_map = dict()
        self.data_lock = Lock()
        self.barrier = None
        self.locations_locks = None

        self.thread = DeviceThread(self)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            self.locations_locks = dict()

        for device in devices:
            if device.device_id != self.device_id:
                if self.device_id == 0:
                    device.locations_locks = self.locations_locks
                    device.barrier = self.barrier

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

            
            for (script, location) in self.device.scripts:
                
                if location not in self.device.locations_locks:
                    self.device.locations_locks[location] = Lock()

                
                self.device.locations_locks[location].acquire()

                script_data = []
                
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
                        device.data_lock.acquire()
                        device.set_data(location, result)
                        device.data_lock.release()
                    
                    self.device.data_lock.acquire()
                    self.device.set_data(location, result)
                    self.device.data_lock.release()

                self.device.locations_locks[location].release()

            
            self.device.barrier.wait()
            self.device.script_received.clear()
