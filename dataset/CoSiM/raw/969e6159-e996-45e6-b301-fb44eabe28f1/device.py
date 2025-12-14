


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

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        flag = True
        device_number = len(devices)

        
        for dev in devices:
            if self.device_id > dev.device_id:
                flag = False

        if flag == True:
            barrier = ReusableBarrierSem(device_number)
            map_locations = {}
            tmp = {}
            for dev in devices:
                dev.barrier = barrier
                tmp = list(set(dev.sensor_data) - set(map_locations))
                for i in tmp:
                    map_locations[i] = Lock()
                dev.map_locations = map_locations
                tmp = {}

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
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
        
        Thread.__init__(self)
        self.device = device

    def run(self):
        
        while True:
            
            self.device.timepoint_done.clear()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.device.timepoint_done.wait()
            script_list = []
            thread_list = []
            index = 0
            for script in self.device.scripts:
                script_list.append(script)
            for i in xrange(8):
                thread = SingleDeviceThread(self.device, script_list, neighbours, index)


                thread.start()
                thread_list.append(thread)
            for i in xrange(len(thread_list)):
                thread_list[i].join()
            self.device.barrier.wait()

class SingleDeviceThread(Thread):
    
    def __init__(self, device, script_list, neighbours, index):
        Thread.__init__(self)
        self.device = device
        self.script_list = script_list
        self.neighbours = neighbours
        self.index = index

    def run(self):
      
        if self.script_list != []:
            (script, location) = self.script_list.pop(self.index)
            self.compute(script, location)

    def update(self, result, location):
        
        for device in self.neighbours:
            device.set_data(location, result)
        self.device.set_data(location, result)

    def collect(self, location, neighbours, script_data):
        
        self.device.map_locations[location].acquire()
        for device in self.neighbours:
            
            data = device.get_data(location)
            if data is None:
                pass
            else:
                script_data.append(data)

        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

    def compute(self, script, location):
        
        script_data = []
        self.collect(location, self.neighbours, script_data)

        if script_data == []:
            pass
        else:
            
            result = script.run(script_data)
            self.update(result, location)

        self.device.map_locations[location].release()

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