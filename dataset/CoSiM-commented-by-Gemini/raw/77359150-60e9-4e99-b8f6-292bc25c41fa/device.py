


from threading import Event, Thread, Lock
import barrier
import runner


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barr = None
        self.devices = []
        self.runners = []
        self.locks = [None] * 50
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        
        if self.barr is None:
            barr = barrier.ReusableBarrierSem(len(devices))
            self.barr = barr
            for dev in devices:
                if dev.barr is None:
                    dev.barr = barr
        
        for dev in devices:
            if dev is not None:
                self.devices.append(dev)

    def assign_script(self, script, location):
        
        
        
        
        ok = 0
        if script is not None:
            self.scripts.append((script, location))
            
            
            
            if self.locks[location] is None:
                for device in self.devices:
                    if device.locks[location] is not None:
                        self.locks[location] = device.locks[location]
                        ok = 1
                        break
                if ok == 0:
                    self.locks[location] = Lock()
            self.script_received.set()
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

    def run(self):
        
        
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                run = runner.ScriptRunner(self.device, script, location,
                                          neighbours)
                self.device.runners.append(run)

                n = len(self.device.runners)
                x = n / 8
                r = n % 7
                
                
                self.device.locks[location].acquire()
                for i in xrange(0, x):
                    for j in xrange(0, 8):
                        self.device.runners[i * 8 + j].start()
                
                
                if n >= 8:
                    for i in xrange(len(self.device.runners) - r,
                                    len(self.device.runners)):
                        self.device.runners[i].start()
                
                
                else:
                    for i in xrange(0, n):
                        self.device.runners[i].start()
                for i in xrange(0, n):
                    self.device.runners[i].join()
                
                self.device.locks[location].release()
                
                self.device.runners = []

            
            self.device.timepoint_done.clear()
            
            self.device.barr.wait()


from threading import Thread


class ScriptRunner(Thread):
    

    def __init__(self, device, script, location, neighbours):
        
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        
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
            
            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            
            self.device.set_data(self.location, result)
