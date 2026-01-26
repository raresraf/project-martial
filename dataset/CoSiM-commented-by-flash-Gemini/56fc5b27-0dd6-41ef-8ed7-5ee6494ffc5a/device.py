


from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.barrier = None
        self.threads = []
        self.semaphore = Semaphore(8)
        self.lock = {}
        self.all_devices = []
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if len(self.all_devices) == 0:
            self.all_devices = devices

        if self.barrier is None:
            barrier = ReusableBarrierSem(len(devices))
            for device in devices:
                device.barrier = barrier

    def update_locks(self, scripts):
        
        for (_, location) in scripts:
            if not self.lock.has_key(location):
                self.lock[location] = Lock()
                for device in self.all_devices:
                    device.lock[location] = self.lock[location]


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()


    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

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

            


            self.device.update_locks(self.device.scripts)

            
            self.device.barrier.wait()

            
            for script in self.device.scripts:
                thread = ScriptThread(self.device, neighbours, script)
                self.device.threads.append(thread)

                
                self.device.semaphore.acquire()
                thread.start()


            for thread in self.device.threads:
                thread.join()

            self.device.threads = []

            
            self.device.script_received.clear()

            
            self.device.barrier.wait()

class ScriptThread(Thread):
    

    def __init__(self, device, neighbours, script):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script

    def run(self):

        
        self.device.lock.get(self.script[1]).acquire()

        script_data = []
        


        for device in self.neighbours:
            data = device.get_data(self.script[1])
            if data is not None:
                script_data.append(data)

        
        data = self.device.get_data(self.script[1])
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script[0].run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.script[1], result)
                
            self.device.set_data(self.script[1], result)

        
        self.device.lock.get(self.script[1]).release()

        
        self.device.semaphore.release()
