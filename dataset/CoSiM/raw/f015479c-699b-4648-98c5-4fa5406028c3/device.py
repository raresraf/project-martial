


from threading import Thread, Lock, Semaphore, Event

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
    

    def __init__(self, device_id, sensor_data, supervisor):
        

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.lock_neigh = None
        self.lock_mine = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id



    def setup_devices(self, devices):
        
        no_devices = len(devices)
        lock_neigh = Lock()
        barrier = ReusableBarrierSem(no_devices)

        
        if self.device_id == 0:
            for i in range(no_devices):
                devices[i].barrier = barrier
                devices[i].lock_neigh = lock_neigh


    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            
            self.script_received.set()
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


class WorkerThread(Thread):

    

    def __init__(self, device, script, location, neighbours):
        

        Thread.__init__(self, name="Worker Thread")
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def collect_data(self, location_data):
        
        location_data.append(self.device.get_data(self.location))
        for i in range(len(self.neighbours)):
            data = self.neighbours[i].get_data(self.location)
            location_data.append(data)

    def update_neighbours(self, result):
        
        no_neigh = len(self.neighbours)
        for i in range(no_neigh):
            self.device.lock_neigh.acquire()
            value = self.neighbours[i].get_data(self.location)
            self.neighbours[i].set_data(self.location, max(result, value))
            self.device.lock_neigh.release()

    def update_self(self, result):
        
        self.device.lock_mine.acquire()
        value = self.device.get_data(self.location)
        self.device.set_data(self.location, max(result, value))
        self.device.lock_mine.release()

    def run(self):
        location_data = []
        self.collect_data(location_data)

        if len(location_data) > 0:
            result = self.script.run(location_data)
            self.update_neighbours(result)
            self.update_self(result)


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        threads = [None] * 200
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()
            self.device.script_received.clear()
            for i in range(len(self.device.scripts)):
                (script, location) = self.device.scripts[i]
                threads[i] = WorkerThread(self.device, script, \
                    location, neighbours)
                threads[i].start()

            for i in range(len(self.device.scripts)):
                threads[i].join()

            self.device.barrier.wait()
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
