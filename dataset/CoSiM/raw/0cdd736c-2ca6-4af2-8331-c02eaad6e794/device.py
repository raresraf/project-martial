

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.devices_list = []
        self.slavet_started = []

        
        self.locks_ready = Event()
        
        self.master_barrier = Event()
        self.lock = Lock()

        self.master_id = None
        self.barrier = None
        self.data_lock = [None] * 100
        self.master_node = True

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
        
        
        for index in range(len(devices)):
            if devices[index] is not None:
                if devices[index].master_id is not None:
                    self.master_id = devices[index].master_id
                    self.master_node = False
                    break
        if not self.master_node:
            for index in range(len(devices)):
                if devices[index] is not None:
                    if devices[index].device_id == self.master_id:
                        devices[index].master_barrier.wait()
                        if self.barrier is None:
                            self.barrier = devices[index].barrier
                    self.devices_list.append(devices[index])
        else:
            self.barrier = ReusableBarrierSem(len(devices))
            self.master_id = self.device_id
            i = 0
            while i < 100:
                self.data_lock[i] = Lock()
                i += 1
            self.locks_ready.set()
            self.master_barrier.set()
            for index in range(len(devices)):
                if devices[index] is not None:
                    devices[index].barrier = self.barrier
                    self.devices_list.append(devices[index])

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))

            for index in range(len(self.devices_list)):
                if self.devices_list[index].device_id == self.master_id:
                    self.devices_list[index].locks_ready.wait()
            for index in range(len(self.devices_list)):
                if self.devices_list[index].device_id == self.master_id:
                    self.data_lock = self.devices_list[index].data_lock

            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

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
                self.device.slavet_started.append(SlaveThread(self.device,
                                                            neighbours,
                                                            location,
                                                            script))
                index = len(self.device.slavet_started) - 1
                self.device.slavet_started[index].start()

            for index in range(len(self.device.slavet_started)):
                self.device.slavet_started[index].join()

            
            self.device.timepoint_done.clear()

            self.device.barrier.wait()


class SlaveThread(Thread):
    

    def __init__(self, device, neighbours, location, script):
        Thread.__init__(self, name="Slave Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.location = location
        self.script = script

    def run(self):
        self.device.data_lock[self.location].acquire()

        if self.neighbours is not None:
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

            self.device.data_lock[self.location].release()
