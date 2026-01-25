


from threading import Thread, Lock
from barrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.thread = DeviceThreadPool(self)
        
        self.barrier = None
        
        self.inner_barrier = ReusableBarrier(2)
        
        self.lock = None
        
        self.inner_lock = Lock()
        
        
        self.lock_map = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        device_ids = [device.device_id for device in devices]
        leader_id = min(device_ids)

        
        if self.device_id == leader_id:
            barrier = ReusableBarrier(len(devices))
            lock = Lock()
            lock_map = {}
            for device in devices:
                device.set_barrier(barrier)
                device.set_lock(lock)
                device.set_lock_map(lock_map)
                device.thread.start()

    def set_barrier(self, barrier):
        
        self.barrier = barrier

    def set_lock(self, lock):
        
        self.lock = lock

    def set_lock_map(self, lock_map):
        
        self.lock_map = lock_map

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))

            with self.lock:


                if location not in self.lock_map:
                    self.lock_map[location] = Lock()
        else:
            self.inner_barrier.wait()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            with self.inner_lock:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThreadPool(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

        while True:
            
            with self.device.lock:
                neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break

            
            self.device.inner_barrier.wait()

            threads = []

            
            for (script, location) in self.device.scripts:
                thread = DeviceThread(self.device, script, location, neighbours)
                thread.start()
                threads.append(thread)

            
            for thread in threads:
                thread.join()

            
            self.device.barrier.wait()


class DeviceThread(Thread):
    

    def __init__(self, device, script, location, neighbours):

        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):

        
        with self.device.lock_map[self.location]:

            script_data = []
            
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if len(script_data) != 0:
                
                result = self.script.run(script_data)

                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)
