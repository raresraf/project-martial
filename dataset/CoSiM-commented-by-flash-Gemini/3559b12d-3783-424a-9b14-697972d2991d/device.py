


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

        self.location_locks = {} 
        self.script_locks = {}
        self.barrier = None

        self.threads = []
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        barrier = ReusableBarrierSem(len(devices))
        for device in devices:
            device.barrier = barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()
        for thread in self.threads:
            thread.join()


class DeviceThreadHelper(Thread):
    

    def __init__(self, device, script, location, neighbours):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script


    def run(self):
        script_data = []

        if self.location not in self.device.location_locks:
            self.device.location_locks[self.location] = Lock()

        if self.script not in self.device.script_locks:
            self.device.script_locks[self.script] = Lock()

        self.device.location_locks[self.location].acquire()

        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        if script_data != []:
            
            self.device.script_locks[self.script].acquire()
            result = self.script.run(script_data)
            self.device.script_locks[self.script].release()

            
            for device in self.neighbours:
                device.set_data(self.location, result)

        self.device.location_locks[self.location].release()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break 


            neighbours.append(self.device) 

            self.device.timepoint_done.wait() 

            for (script, location) in self.device.scripts:
                thread = DeviceThreadHelper(self.device, script, location, neighbours)
                self.device.threads.append(thread)
                thread.start()

            for thread in self.device.threads:
                thread.join()

            self.device.threads = []

            self.device.barrier.wait()
            self.device.timepoint_done.clear()
