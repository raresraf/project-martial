


from threading import Event, Thread, Lock, Semaphore

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor


        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.max_threads = 8
        self.barrier = None
        self.location_locks = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        barrier = Barrier(len(devices))
        location_locks = {}

        for device in devices:
            device.barrier = barrier
            device.location_locks = location_locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
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


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            
            scripts_to_threads = [self.device.scripts[i::self.device.max_threads]
                                  for i in range(self.device.max_threads)]
            threads = []

            
            for scripts in scripts_to_threads:
                if scripts != []:
                    thread = Thread(target=self.run_thread, args=(neighbours, scripts))
                    thread.start()
                    threads.append(thread)

            
            for thread in threads:
                thread.join()

            
            self.device.barrier.wait()
            self.device.timepoint_done.clear()


    def run_thread(self, neighbours, scripts):
        

        for (script, location) in scripts:

            if location not in self.device.location_locks:
                self.device.location_locks[location] = Lock()
            self.device.location_locks[location].acquire()

            
            data = [dev.get_data(location) for dev in neighbours if dev.get_data(location)]
            if self.device.get_data(location):
                data += [self.device.get_data(location)]

            
            if data != []:
                result = script.run(data)
                self.device.set_data(location, result)
                for neighbour in neighbours:
                    neighbour.set_data(location, result)

            self.device.location_locks[location].release()

class Barrier(object):
    

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
