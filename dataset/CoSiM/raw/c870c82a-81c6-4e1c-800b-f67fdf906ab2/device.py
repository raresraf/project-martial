


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
        
        self.device_id = device_id


        self.sensor_data = sensor_data
        self.lock = {}

        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_done = Event()
        self.terminate = Event()
        self.neighbours = []

        self.barrier = None
        self.threads_barrier = ReusableBarrierSem(9)
        self.master = DeviceThread(self, self.terminate, self.barrier, self.threads_barrier, \
                                    self.setup_done)
        self.master.start()

        self.threads = []

        for _ in range(8):
            thread = Worker(self.master, self.terminate, self.threads_barrier)

            self.threads.append(thread)
            thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                self.lock[dev] = Lock()
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier
                    dev.lock = self.lock
                    dev.setup_done.set()

            self.setup_done.set()

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
        

        self.terminate.set()
        for i in range(8):
            self.threads[i].script_received.set()
            self.threads[i].join()
        self.master.join()


class DeviceThread(Thread):
    

    def __init__(self, device, terminate, barrier, threads_barrier, setup_done):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = []
        self.terminate = terminate
        self.barrier = barrier
        self.threads_barrier = threads_barrier
        self.setup_done = setup_done

    def run(self):

        
        self.setup_done.wait()
        self.device.barrier.wait()

        while True:
            
            self.device.barrier.wait()

            


            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                break

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

            
            scripts = []
            for i in range(8):
                scripts.append([])

            for i in range(len(self.device.scripts)):
                scripts[i%8].append(self.device.scripts[i])

            for i in range(8):
                self.device.threads[i].scripts = scripts[i]
                self.device.threads[i].script_received.set()

            
            if not self.terminate.is_set():
                self.threads_barrier.wait()


class Worker(Thread):
    

    def __init__(self, master, terminate, barrier):

        Thread.__init__(self)
        self.master = master
        self.script_received = Event()
        self.terminate = terminate
        self.scripts = []
        self.barrier = barrier

    @staticmethod
    def append_data(device, location, script_data):
        
        device.lock[device].acquire()
        data = device.get_data(location)
        device.lock[device].release()
        if data is not None:
            script_data.append(data)

    @staticmethod
    def set_data(device, location, result):
        
        device.lock[device].acquire()
        device.set_data(location, result)
        device.lock[device].release()

    def run(self):

        while True:
            self.script_received.wait()
            self.script_received.clear()

            if self.terminate.is_set():
                break
            if self.scripts is not None:
                for (script, location) in self.scripts:

                    
                    script_data = []
                    if self.master.neighbours is not None:
                        
                        for device in self.master.neighbours:
                            self.append_data(device, location, script_data)


                    
                    self.append_data(self.master.device, location, script_data)

                    if script_data != []:

                        result = script.run(script_data)

                        if self.master.neighbours is not None:
                            
                            for device in self.master.neighbours:
                                self.set_data(device, location, result)
                        
                        self.set_data(self.master.device, location, result)

            self.barrier.wait()
