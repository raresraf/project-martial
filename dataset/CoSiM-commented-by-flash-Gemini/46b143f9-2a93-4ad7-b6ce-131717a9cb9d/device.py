


from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.scripts_lock = Lock()
        self.timepoint_done = Event()
        self.barrier = None
        self.location_locks = {}
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        barrier = ReusableBarrierCond(len(devices))
        for device in devices:
            device.barrier = barrier

        location_locks = {}

        for device in devices:
            for location in device.sensor_data:
                if location not in location_locks:
                    location_locks[location] = Lock()

        for device in devices:
            device.location_locks = location_locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts_lock.acquire()
            self.scripts.append((script, location))
            self.scripts_lock.release()
            self.script_received.set()


        else:
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def process_work(self, script, location, neighbours):
        
        self.location_locks[location].acquire()

        script_data = []

        
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        
        data = self.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = script.run(script_data)

            for device in neighbours:
                device.set_data(location, result)

            self.set_data(location, result)

        self.location_locks[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        work_lock = Lock()
        work_pool_empty = Event()
        work_pool_empty.set()
        work_pool = []
        workers = []
        workers_number = 7
        work_available = Semaphore(0)
        own_work = None

        for worker_id in range(1, workers_number + 1):
            workers.append(Worker(worker_id, work_pool, work_available, work_pool_empty, work_lock, self.device))
            workers[worker_id-1].start()

        while True:
            scripts_ran = []
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is not None:
                
                neighbours = set(neighbours)

                if self.device in neighbours:
                    neighbours.remove(self.device)

            if neighbours is None:
                
                for i in range(0,7):
                    work_available.release()

                
                for worker in workers:
                    worker.join()
                break

            
            self.device.barrier.wait()

            while True:
                
                self.device.script_received.wait()

                self.device.scripts_lock.acquire()

                
                for (script, location) in self.device.scripts:
                    
                    if script in scripts_ran:
                        continue

                    
                    scripts_ran.append(script)

                    
                    if own_work is None:
                        own_work = (script, location, neighbours)
                    
                    else:
                        work_lock.acquire()
                        work_pool.append((script, location, neighbours))
                        work_pool_empty.clear()
                        work_available.release()
                        work_lock.release()

                self.device.scripts_lock.release()

                
                if self.device.timepoint_done.is_set() and len(scripts_ran) == len(self.device.scripts):
                    
                    if own_work is not None:
                        script, location, neighbours = own_work
                        own_work = None
                        self.device.process_work(script, location, neighbours)

                    
                    work_pool_empty.wait()

                    
                    for worker in workers:
                        worker.work_done.wait()

                    self.device.timepoint_done.clear()
                    
                    self.device.barrier.wait()
                    break


class Worker(Thread):
    

    def __init__(self, worker_id, work_pool, work_available, work_pool_empty, work_lock, device):
        
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.work_pool = work_pool
        self.work_available = work_available
        self.work_pool_empty = work_pool_empty
        self.work_lock = work_lock
        self.device = device
        self.work_done = Event()
        self.work_done.set()

    def run(self):

        while True:
            self.work_available.acquire()
            self.work_lock.acquire()
            self.work_done.clear()

            
            if not self.work_pool:
                self.work_lock.release()
                return

            
            script, location, neighbours = self.work_pool.pop(0)

            
            if not self.work_pool:
                self.work_pool_empty.set()

            self.work_lock.release()

            self.device.process_work(script, location, neighbours)

            self.work_done.set()
