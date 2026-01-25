


from threading import Event, Thread, Lock, Semaphore, RLock


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
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        


        self.devices = []
        self.script_received = Event()
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        for device in devices:
            self.devices.append(device)
        self.devices[0].barrier = ReusableBarrier(len(self.devices))
        self.devices[0].locations_lock = {}

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.num_threads = 8

    def run(self):
        
        while True:
            
            threads = []
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.script_received.wait()

            
            for (script, location) in self.device.scripts:
                thread = MyThread(self, script, location, neighbours)
                threads.append(thread)

            rounds = len(self.device.scripts) / self.num_threads
            leftovers = len(self.device.scripts) % self.num_threads
            
            while rounds > 0:
                for j in xrange(self.num_threads):
                    threads[j].start()
                for j in xrange(self.num_threads):
                    threads[j].join()
                for j in xrange(self.num_threads):
                    threads.pop(0)
                rounds -= 1
            
            for j in xrange(leftovers):
                threads[j].start()
            for j in xrange(leftovers):
                threads[j].join()
            for j in xrange(leftovers):
                threads.pop(0)

            
            
            del threads[:]
            
            self.device.devices[0].barrier.wait()
            
            self.device.script_received.clear()


class MyThread(Thread):
    

    def __init__(self, device_thread, script, location, neighbours):
        
        Thread.__init__(self)
        self.device_thread = device_thread
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        
        
        if self.location not in\
                self.device_thread.device.devices[0].locations_lock:
            self.device_thread.device.devices[0].locations_lock[self.location]\
                = RLock()
        with self.device_thread.device.devices[0].locations_lock[self.location]:
            script_data = []
            
            for device in self.neighbours:


                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device_thread.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            if script_data != []:


                result = self.script.run(script_data)
                
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.device_thread.device.set_data(self.location, result)
