


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.devices = None
        self.barrier = None
        self.thread = DeviceThread(self)
        self.locations = []
        self.data_lock = Lock()
        self.get_lock = Lock()
        self.setup = Event()
        self.thread.start()
    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        self.devices = devices
        barrier = ReusableBarrierSem(len(devices))
        if self.device_id == 0:
            for _ in range(100):
                self.locations.append(Lock())


            for dev in devices:
                dev.barrier = barrier
                dev.locations = self.locations
                dev.setup.set()
    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        
        with self.get_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        self.device.setup.wait()
        while True:
            threads = []
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.device.script_received.wait()
            self.device.script_received.clear()
            i = 0
            for _ in self.device.scripts:
                threads.append(MyThread(self.device, self.device.scripts, neighbours, i))
                i = i + 1
            scripts_rem = len(self.device.scripts)
            start = 0
            if len(self.device.scripts) < 8:
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            else:
                while True:
                    if scripts_rem == 0:
                        break
                    if scripts_rem >= 8:


                        for i in xrange(start, start + 8):
                            threads[i].start()
                        for i in xrange(start, start + 8):
                            threads[i].join()
                        start = start + 8
                        scripts_rem = scripts_rem - 8
                    else:
                        for i in xrange(start, start + scripts_rem):
                            threads[i].start()
                        for i in xrange(start, start + scripts_rem):
                            threads[i].join()
                        break
            self.device.barrier.wait()

class MyThread(Thread):
    
    def __init__(self, device, scripts, neighbours, indice):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours
        self.indice = indice

    def run(self):
        
        (script, location) = self.scripts[self.indice]
        self.device.locations[location].acquire()
        script_data = []
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        if script_data != []:
            result = script.run(script_data)
            for device in self.neighbours:
                device.set_data(location, result)
                self.device.set_data(location, result)
        self.device.locations[location].release()

