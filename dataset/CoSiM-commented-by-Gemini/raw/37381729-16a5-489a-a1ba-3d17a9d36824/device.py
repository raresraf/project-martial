


from threading import Event, Thread, Lock
from reusableBarrier import ReusableBarrier

class MyThread(Thread):
    def __init__(self, d, location, script, neighbours):
        Thread.__init__(self)
        self.d = d
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        script_data = []

        for device in self.neighbours:
            keys = device.dictionar.keys()
            if self.location in keys:
                lock = device.dictionar[self.location]
                if lock is not None:
                    lock.acquire()


            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.d.device.get_data(self.location)
        if data is not None:
            script_data.append(data)
        if script_data != []:
            

            result = self.script.run(script_data)
            
            for device in self.neighbours:
                device.set_data(self.location, result)
                keys = device.dictionar.keys()
                if self.location in keys:
                    lock = device.dictionar[self.location]
                    if lock is not None:
                        lock.release()
            
            self.d.device.lock.acquire()
            self.d.device.set_data(self.location, result)
            self.d.device.lock.release()

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
        self.lock = Lock()
        self.dictionar = {}
        for location in self.sensor_data:
            if location != None:
                self.dictionar[location] = Lock()
            else:
                self.dictionar[location] = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
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


            self.device.timepoint_done.clear()


            my_thread_list = []
            
            for (script, location) in self.device.scripts:
                my_thread = MyThread(self, location, script, neighbours)
                my_thread_list.append(my_thread)
                my_thread.start()
            for thread in my_thread_list:
                thread.join()

            
            self.device.barrier.wait()

