


from threading import Event, Thread, Lock
from multiprocessing.dummy import Pool as ThreadPool
from reusablebarrier import ReusableBarrierCond

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor


        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.data_locks = {}
        for location in sensor_data:
            self.data_locks[location] = Lock()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            self.data_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_locks[location].release()

    def shutdown(self):
        
        self.thread.join()

class Helper(object):
    
    def __init__(self, device):
        
        self.device = device
        self.pool = ThreadPool(8)
        self.neighbours = None
        self.scripts = None

    def set_neighbours_and_scripts(self, neighbours, scripts):
        
        self.neighbours = neighbours
        self.scripts = scripts

    def script_run(self, (script, location)):
        
        script_data = []
        for device in self.neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        if script_data != []:
            result = script.run(script_data)
            for device in self.neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)
            self.device.set_data(location, result)

    def run(self):
        
        self.pool.map_async(self.script_run, self.scripts)

    def close_pool(self):
        
        self.pool.close()
        self.pool.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.helper = None

    def run(self):



        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.helper = Helper(self.device)
            
            while True:
                if (self.device.script_received.is_set() or
                self.device.timepoint_done.is_set()):
                    
                    
                    
                    if self.device.script_received.is_set():
                        self.device.script_received.clear()
                        self.helper.set_neighbours_and_scripts(neighbours,
							self.device.scripts)
                        self.helper.run()
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break
            self.helper.close_pool()
            self.device.barrier.wait()
