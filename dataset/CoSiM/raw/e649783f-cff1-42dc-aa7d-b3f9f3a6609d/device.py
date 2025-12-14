


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
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.threads = []
        self.devices = []
        self.semafor = Semaphore(0)
        self.timepoint_done = Event()
        self.thread = SupervisorThread(self)
        self.thread.start()
        self.num_scr = 8
        self.lock = [None] * 100

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.barrier is None:
            self.barrier = ReusableBarrier(len(devices))
            for dev in devices:
                dev.lock = self.lock
                if dev.barrier is None:


                    dev.barrier = self.barrier

        for dev in devices:
            if dev is not None:
                self.devices.append(dev)

    def assign_script(self, script, location):
        


        if script is not None:
            self.scripts.append((script, location))
            if self.lock[location] is None:
                for device in self.devices:
                    if device.lock[location] is not None:
                        self.lock[location] = device.lock[location]
                        break
                    self.lock[location] = Lock()
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


class SupervisorThread(Thread):
    
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            neighb = self.device.supervisor.get_neighbours()
            if neighb is None:
                break
            self.device.timepoint_done.wait()
            i = 0
            while i < len(self.device.scripts):

                for _ in range(0, self.device.num_scr):
                    pair = self.device.scripts[i]
                    new_thread = Slave(self.device, pair[1], neighb, pair[0])
                    self.device.threads.append(new_thread)
                    new_thread.start()
                    i = i + 1
                    if i >= len(self.device.scripts):
                        break
                for thread in self.device.threads:
                    thread.join()

            self.device.threads = []
            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

class Slave(Thread):
    

    def __init__(self, device, location, neighbours, script):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script


    def run(self):
        self.device.lock[self.location].acquire()
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
        self.device.lock[self.location].release()
