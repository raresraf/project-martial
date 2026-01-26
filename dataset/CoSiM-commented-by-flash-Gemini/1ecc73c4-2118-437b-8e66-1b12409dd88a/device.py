


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
        self.start_event = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.data_lock = {}

        
        for data in sensor_data:
            self.data_lock[data] = Lock()

        self.barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.barrier == None:
            self.barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                dev.barrier = self.barrier

        self.start_event.set()

    def assign_script(self, script, location):
        

        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

        self.script_received.set()

    def get_data(self, location):
        

        
        if location in self.sensor_data:
            self.data_lock[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        

        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_lock[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceCore(Thread):
    
    def __init__(self, device, location, script, neighbours):
        
        Thread.__init__(self, name="Device core %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        script_data = []
        
        for device in self.neighbours:
            if self.device.device_id != device.device_id:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                if self.device.device_id != device.device_id:


                    device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        self.device.start_event.wait()

        while True:
            self.device.barrier.wait()

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            while not self.device.timepoint_done.is_set():
                self.device.script_received.wait()

            
            used_cores = 0
            free_core = list(range(8))
            threads = {}

            
            
            
            
            

            for (script, location) in self.device.scripts:
                if used_cores < 8:
                    dev_core = DeviceCore(self.device, location, script, neighbours)
                    dev_core.start()
                    threads[free_core.pop()] = dev_core
                    used_cores = used_cores + 1

                else:
                    for thread in threads:
                        if not threads[thread].isAlive():
                            threads[thread].join()
                            free_core.append(thread)
                            used_cores = used_cores - 1

            for thread in threads:
                threads[thread].join()

            
            self.device.timepoint_done.clear()
            if self.device.script_received.is_set():
                self.device.script_received.clear()
