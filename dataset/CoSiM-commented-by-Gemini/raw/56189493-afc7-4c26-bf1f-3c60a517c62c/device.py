


from threading import Lock, Event, Thread, Semaphore, Condition


class ReusableBarrierSem():


    
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
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    
    location_locks = []
    barrier = None
    nr_t = 8
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.neighbours_event = Event()
        self.threads = []
        for i in xrange(Device.nr_t):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(Device.nr_t):
            self.threads[i].start()
    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        Device.barrier = ReusableBarrierSem(Device.nr_t * len(devices))

    def assign_script(self, script, location):
        
        if location not in [elem[0] for elem in Device.location_locks]:
            Device.location_locks.append((location, Lock()))

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
        
        for i in xrange(Device.nr_t):
            self.threads[i].join()

class DeviceThread(Thread):

    

    def __init__(self, device, index):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.index = index
        self.neighbours = None

    def run(self):
        
        
        while True:
            
            
            
            
            if self.index == 0:
                self.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_event.set()
            else:
                self.device.neighbours_event.wait()
                self.neighbours = self.device.threads[0].neighbours
            if self.neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            
            for j in range(self.index, len(self.device.scripts), Device.nr_t):
                location = self.device.scripts[j][1]
                script = self.device.scripts[j][0]

                
                for i in range(len(Device.location_locks)):
                    if location == Device.location_locks[i][0]:
                        Device.location_locks[i][1].acquire()

                script_data = []
                
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    
                    for device in self.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                for i in range(len(Device.location_locks)):
                    if location == Device.location_locks[i][0]:
                        Device.location_locks[i][1].release()

            Device.barrier.wait()
            if self.index == 0:
                self.device.timepoint_done.clear()

            if self.index == 0:
                self.device.neighbours_event.clear()
            Device.barrier.wait()

