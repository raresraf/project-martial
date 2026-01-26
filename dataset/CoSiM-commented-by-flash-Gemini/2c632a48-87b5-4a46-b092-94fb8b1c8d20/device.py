


from threading import Event, Thread, Condition, RLock


class ReusableBarrier(object):
    

    def __init__(self, num_threads):
        
        
        self.num_threads = num_threads
         
        self.count_threads = self.num_threads
        
        self.cond = Condition()

    def wait(self):
        
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
        
            self.cond.notify_all()
            
            self.count_threads = self.num_threads
        else:
        
            self.cond.wait()
        
        self.cond.release()


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.can_be_write = RLock()
        self.scripts = []
        self.devices = []


        self.locationl = []
        self.locations = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.devices = devices
        
        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        else:
            self.barrier = self.devices[0].barrier

        
        
        if self.device_id != 0:
            self.locationl = self.devices[0].locationl
            self.locations = self.devices[0].locations

        
        for loc in self.sensor_data:
            self.locationl.append((loc, RLock()))


            self.locations.append(loc)

    def assign_script(self, script, location):
        
        
        if location not in self.locations:
            for dev in self.devices:
                dev.locationl.append((location, RLock()))
                dev.locations.append(location)
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in\
                                             self.sensor_data else None

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
            self.device.script_received.wait()
            self.device.script_received.clear()
            threads = []
            i = 1
            
            
            
            
            
            for (script, location) in self.device.scripts:
                threads.append(ScriptThread(self.device, neighbours,
                                            location, script))
                if i % 8 == 0:
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    threads = []
                i = i+1
            
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.device.barrier.wait()
            self.device.timepoint_done.wait()

class ScriptThread(Thread):
    

    def __init__(self, device, neighbours, location, script):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.location = location
        self.script = script

    def run(self):
        
        
        for (loc, lock) in self.device.locationl:
            if loc == self.location:
                lock.acquire()
        
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
        
        for (loc, lock) in self.device.locationl:
            if loc == self.location:
                lock.release()
