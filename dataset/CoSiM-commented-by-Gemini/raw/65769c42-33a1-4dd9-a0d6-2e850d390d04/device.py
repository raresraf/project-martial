


from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier


class Device(object):
    
    timepoint_barrier = None
    script_lock = None
    data_lock = None
    data_locks = {}

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.devices = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.devices = devices
        num_devices = len(devices)
        if self.device_id == 0:
            self.timepoint_barrier = ReusableBarrier(num_devices)
            self.script_lock = Lock()
            self.data_lock = Lock()
            for i in range(1, len(devices)):
                devices[i].data_lock = self.data_lock
                devices[i].script_lock = self.script_lock
                devices[i].timepoint_barrier = self.timepoint_barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
            if not location in self.data_locks:
                lock = Lock()


                for dev in self.devices:
                    dev.data_locks[location] = lock
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script_threads = []

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                script_thread = ScriptThread(self.device, script, location, \
                    neighbours)
                script_thread.start()
                self.script_threads.append(script_thread)

            
            self.device.timepoint_done.clear()
            for script_thread in self.script_threads:
                script_thread.join()
            self.script_threads = []
            self.device.timepoint_barrier.wait()

class ScriptThread(Thread):
    

    def __init__(self, device, script, location, neighbours):
        
        Thread.__init__(self, name="Script Thread %d" % device.device_id)


        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        with self.device.data_locks[self.location]:
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

                with self.device.script_lock:
                    
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    
                    self.device.set_data(self.location, result)
                    