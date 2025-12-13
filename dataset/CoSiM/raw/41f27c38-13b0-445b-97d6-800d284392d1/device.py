


from threading import Event, Thread, Semaphore
from concurrent.futures import wait, ALL_COMPLETED, ThreadPoolExecutor
from barrier import ReusableBarrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = ReusableBarrier(1)
        self.thread = DeviceThread(self)
        self.location_sems = {location : Semaphore(1) for location in sensor_data}
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for dev in devices:
                dev.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            self.location_sems[location].acquire()
            data = self.sensor_data[location]
        else:
            data = None
        return data

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_sems[location].release()

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPoolExecutor(8)
        self.neighbours = []

    def gather_info(self, location):
        
        script_data = []
        for device in self.neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        return script_data

    def spread_info(self, result, location):
        


        for device in self.neighbours:
            if device.device_id != self.device.device_id:
                device.set_data(location, result)
        self.device.set_data(location, result)

    def update(self, script, location):
        
        script_data = self.gather_info(location)
        result = None
        if script_data != []:
            result = script.run(script_data)
            self.spread_info(result, location)

    def run(self):
        
        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()

            
            if self.neighbours is None:
                break
            futures = []
            while True:
                
                
                if self.device.script_received.is_set() or self.device.timepoint_done.wait():


                    if self.device.script_received.is_set():
                        
                        self.device.script_received.clear()
                        for (script, location) in self.device.scripts:
                            future = self.thread_pool.submit(self.update, script, location)
                            futures.append(future)
                    else:
                        
                        
                        
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break
            
            wait(futures, timeout=None, return_when=ALL_COMPLETED)
            
            self.device.barrier.wait()
        self.thread_pool.shutdown()
