


from threading import Event, Thread
from barrier import *

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
        self.devices = []
        self.barrier = None
        self.threads = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        

        if self.barrier is None:


            barrier = ReusableBarrierSem(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
        for device in devices:
            if device is not None:
                self.devices.append(device)
    def assign_script(self, script, location):
        
        flag = 0
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

            
            for (script, location) in self.device.scripts:
                mythread = MyThread(self.device, location, script, neighbours)
                self.device.threads.append(mythread)

            for xthread in self.device.threads:
                xthread.start()
            for xthread in self.device.threads:
                xthread.join()

            
            self.device.threads = []
            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
