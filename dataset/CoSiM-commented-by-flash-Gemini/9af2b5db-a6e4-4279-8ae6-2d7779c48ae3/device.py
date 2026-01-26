


from threading import *

class Barrier():
    

    def __init__(self):
        self.threads_num = 0
        self.count1_threads = 0
        self.count2_threads = 0
        self.counter_lock = Lock()
        self.semafor1 = Semaphore(0)
        self.semafor2 = Semaphore(0)

    def init_devices (self, dev_nr):
        self.threads_num = dev_nr
        self.count1_threads = dev_nr
        self.count2_threads = dev_nr

    def fazaI (self):
        with self.counter_lock:
            self.count1_threads -= 1
            if self.count1_threads == 0:
                for i in range (self.threads_num):
                    self.semafor1.release()
                self.count1_threads = self.threads_num
        self.semafor1.acquire()

    def fazaII (self):
        with self.counter_lock:
            self.count2_threads -= 1
            if self.count2_threads == 0:
                for i in range (self.threads_num):
                    self.semafor2.release()
                self.count2_threads = self.threads_num
        self.semafor2.acquire()

    def wait(self):
        self.fazaI()
        self.fazaII()

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

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        DeviceThread.bariera.init_devices(len(devices))

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
    

    
    bariera = Barrier()

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
                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

            
            self.device.timepoint_done.clear()

            DeviceThread.bariera.wait()