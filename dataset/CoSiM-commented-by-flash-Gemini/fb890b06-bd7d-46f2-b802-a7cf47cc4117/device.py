


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
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.devices = []
        self.locks = {}
        self.lock_used = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        barrier = ReusableBarrierSem(len(devices))

        
        for device in devices:
            self.devices.append(device)
            device.barrier = barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            
            for device in self.devices:
                if device.locks.get(location) is not None:
                    self.locks[location] = device.locks[location]
                    self.lock_used = 1
                    break

            
            if self.lock_used is None:
                self.locks[location] = Lock()

            self.lock_used = None
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class ScriptThread(Thread):
    

    def __init__(self, device, script, location, neighbours):
        
        Thread.__init__(self)


        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        
        with self.device.locks[self.location]:
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
                thread = ScriptThread(self.device, script, location, neighbours)
                self.script_threads.append(thread)

            for thread in self.script_threads:
                thread.start()
            for thread in self.script_threads:
                thread.join()
            
            self.script_threads = []

            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
