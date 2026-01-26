


from threading import Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.queue = Queue()
        self.num_threads = 8

        self.location_locks = None
        self.lock = None
        self.barrier = None

        self.thread = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.location_locks = {}
            self.lock = Lock()
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.location_locks = self.location_locks
                    device.lock = self.lock
                    device.barrier = self.barrier
        self.thread = DeviceThread(self)
        self.thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            with self.lock:
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()
            self.queue.put((script, location))
        else:
            for _ in range(self.num_threads):
                self.queue.put((None, None))

    def get_data(self, location):
        
        return self.sensor_data[
            location] if location in self.sensor_data else None

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
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            worker_threads = [WorkerThread(self.device, neighbours) for _ in
                              range(self.device.num_threads)]


            for thread in worker_threads:
                thread.start()
            for thread in worker_threads:
                thread.join()

            self.device.barrier.wait()


class WorkerThread(Thread):
    

    def __init__(self, device, neighbours):
        
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours

    def run_script(self, script, location):
        
        script_data = []
        
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            
            result = script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)

    def run(self):
        
        while True:
            script, location = self.device.queue.get()
            if script is None:
                return
            with self.device.location_locks[location]:
                self.run_script(script, location)
            self.device.queue.put((script, location))
