



from barrier import *
from my_thread import *


class Device(object):
    
    barrier = ReusableBarrier(0)

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.lock = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            Device.barrier = ReusableBarrier(len(devices))

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
        self.threads = []

    def run(self):
        
        while True:
            



            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            self.threads = []
            for i in xrange(8):
                self.threads.append(MyThread("{}-{}".format(self.device.device_id, i)))

            
            
            i = 0
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
                    
                    self.threads[i].add_task(script, location, script_data)
                    i = (i+1) % 8

            for i in xrange(8):
                if self.threads[i].tasks != []:
                    self.threads[i].start()

            for i in xrange(8):
                if self.threads[i].tasks != []:
                    self.threads[i].join()
            
            Device.barrier.wait()

            
            for i in xrange(8):
                if self.threads[i].results != []:
                    for (script, location, result) in self.threads[i].results:
                        for device in neighbours:
                            device.lock.acquire()
                            device.set_data(location, result)
                            device.lock.release()
                        
                        self.device.lock.acquire()
                        self.device.set_data(location, result)
                        self.device.lock.release()

            Device.barrier.wait()
from threading import *


class MyThread(Thread):

    def __init__(self, id):
        Thread.__init__(self, name="Device Thread %s" % id)
        self.tasks = []
        self.results = []

    def add_task(self, script, location, script_data):
        self.tasks.append((script, location, script_data))

    def clear(self):
        self.tasks = []
        self.results = []

    def run(self):
        for (script, location, script_data) in self.tasks:
            self.results.append((script, location, script.run(script_data)))

