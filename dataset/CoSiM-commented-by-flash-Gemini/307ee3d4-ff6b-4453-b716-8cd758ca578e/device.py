


from threading import Event, Thread, Semaphore, Lock

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_scripts = []
        self.neighbours = []
        self.timepoint_done = Event()
        
        self.initialization = Event()
        
        self.threads = []
        for k in xrange(8):
            self.threads.append(DeviceThread(self, k))
        self.locations_lock = Lock()
        self.locked_locations = None
        self.devices_barrier = None
        self.device_barrier = ReusableBarrier(len(self.threads))

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            
            self.locked_locations = {}

            
            self.devices_barrier = ReusableBarrier(len(devices)*len(self.threads))

            
            for device in devices:
                device.locked_locations = self.locked_locations
                device.devices_barrier = self.devices_barrier


                device.initialization.set()

        else:
            
            self.initialization.wait()

        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        while True:
            
            self.device.devices_barrier.wait()

            
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            


            self.device.device_barrier.wait()
            neighbours = self.device.neighbours
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()
            if self.thread_id == 0:
                self.device.timepoint_scripts = self.device.scripts[:]
            self.device.device_barrier.wait()
            while True:
                
                self.device.locations_lock.acquire()
                if len(self.device.timepoint_scripts) == 0:
                    self.device.locations_lock.release()
                    break
                (script, location) = self.device.timepoint_scripts.pop()

                
                if location not in self.device.locked_locations:


                    self.device.locked_locations[location] = Lock()

                self.device.locked_locations[location].acquire()
                self.device.locations_lock.release()

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
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                
                self.device.locked_locations[location].release()

            
            self.device.device_barrier.wait()
            self.device.timepoint_done.clear()

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
