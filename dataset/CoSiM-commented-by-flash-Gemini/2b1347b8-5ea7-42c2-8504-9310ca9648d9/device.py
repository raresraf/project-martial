


from threading import Event, Thread, Lock, Semaphore, RLock

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
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.got_neighbours = False
        self.neighbours_list = []

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.neighbours_lock = Lock()

        self.nr_devices = 0

        
        
        
        
        self.root_device = None
        self.thread_barrier = None

        self.threads = []
        self.nr_threads = 8

        
        self.locks = {}
        for location in sensor_data:
            self.locks[location] = RLock()

        for i in range(self.nr_threads):
            self.threads.append(DeviceThread(self, i))
            self.threads[i].start()


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        self.nr_devices = len(devices)

        if self.device_id == 0:
            self.root_device = self
            self.thread_barrier = ReusableBarrierSem(self.nr_devices * self.nr_threads)

        if self.device_id != 0:
            for dev in devices:
                if dev.device_id == 0:
                    self.root_device = dev

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
        
        for i in range(self.nr_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        
        while True:
            
            
            
            
            self.device.neighbours_lock.acquire()

            if not self.device.got_neighbours:
                self.device.got_neighbours = True
                neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_list = neighbours
            else:
                neighbours = self.device.neighbours_list

            self.device.neighbours_lock.release()

            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            for i in range(len(self.device.scripts)):
                if i % self.device.nr_threads == self.thread_id:
                    (script, location) = self.device.scripts[i]

                    script_data = []
                    
                    for device in neighbours:


                        if location in device.locks:
                            device.locks[location].acquire()

                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                    
                    if location in self.device.locks:
                        self.device.locks[location].acquire()
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        
                        result = script.run(script_data)

                        
                        for device in neighbours:


                            device.set_data(location, result)

                        
                        self.device.set_data(location, result)

                    if location in self.device.locks:
                        self.device.locks[location].release()
                    for device in neighbours:
                        if location in device.locks:
                            device.locks[location].release()

            
            self.device.root_device.thread_barrier.wait()

            self.device.timepoint_done.clear()
            self.device.got_neighbours = False

            self.device.root_device.thread_barrier.wait()
