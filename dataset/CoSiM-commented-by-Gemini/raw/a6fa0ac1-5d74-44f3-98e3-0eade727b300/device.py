


from threading import Event, Thread, Lock
from barrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.locks = {}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        for key in self.sensor_data:
            self.locks[key] = Lock()

        
        if self.device_id == 0:
            self.thread.barrier = ReusableBarrier(len(devices))

            for device in devices:
                device.thread.barrier = self.thread.barrier

            for device in devices:
                device.thread.start()



    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def has_data(self, location):
        
        return location in self.sensor_data

    def get_data(self, location):
        
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = None
        self.work_pool_lock = Lock()
        self.work_pool_empty = Event()
        self.work_ready = Event()
        self.work_pool = []
        self.simulation_complete = False
        self.work_ready.clear()
        self.work_pool_empty.set()

    def run(self):

        workers = []

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            
            for i in range(len(workers), len(self.device.scripts)):
                if len(workers) < 8:
                    worker = Worker(self.work_pool_empty, self.work_ready, self.work_pool_lock, self)
                    workers.append(worker)
                    worker.start()

            self.barrier.wait()

            
            
            for (script, location) in self.device.scripts:
                script_devices = []
                for device in neighbours:
                    if device.has_data(location):
                        script_devices.append(device)

                if script_devices:
                    if self.device not in script_devices:
                        script_devices.append(self.device)
                    script_devices.sort(key=lambda x: x.device_id, reverse=False)
                    self.work_pool_lock.acquire()
                    self.work_pool.append(Task(script, location, script_devices))
                    self.work_ready.set()
                    self.work_pool_empty.clear()
                    self.work_pool_lock.release()

            
            
            
            self.work_pool_empty.wait()
            for worker in workers:
                worker.work_done.wait()
            self.device.timepoint_done.clear()

        
        self.work_pool_lock.acquire()
        self.simulation_complete = True
        self.work_ready.set()
        self.work_pool_lock.release()

        for worker in workers:
            worker.join()


class Worker(Thread):

    def __init__(self, work_pool_empty, work_ready, work_pool_lock, device_thread):
        Thread.__init__(self, name="Worker Thread")
        self.work_pool_lock = work_pool_lock
        self.work_pool_empty = work_pool_empty
        self.work_ready = work_ready
        self.device_thread = device_thread
        self.work_done = Event()
        self.work_done.set()

    def run(self):
        while True:
            
            self.work_ready.wait()
            if self.device_thread.simulation_complete:
                break

            self.work_pool_lock.acquire()
            
            
            
            if not self.device_thread.work_pool:
                self.work_pool_empty.set()
                if not self.device_thread.simulation_complete:
                    self.work_ready.clear()
                self.work_pool_lock.release()
            else:
                
                self.work_done.clear()
                task = self.device_thread.work_pool.pop(0)
                self.work_pool_lock.release()
                data = []
                for device in task.devices:
                    data.append(device.get_data(task.location))

                result = task.script.run(data)

                for device in task.devices:
                    device.set_data(task.location, result)

                
                
                self.work_done.set()


class Task(object):

    def __init__(self, script, location, devices):
        self.devices = devices
        self.script = script
        self.location = location
