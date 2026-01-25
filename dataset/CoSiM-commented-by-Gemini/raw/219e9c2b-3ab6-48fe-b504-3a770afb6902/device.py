


from threading import Event, Thread, Lock, Condition
from execute_scripts import ExecuteScripts

class ReusableBarrierCond(object):
    
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
        self.scripts = []


        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.locks = {}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            locks = {}
            for device in devices:
                device.barrier = barrier
                device.locks = locks
        for location in self.sensor_data.keys():


            if not location in self.locks:
                self.locks[location] = Lock()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        
        loc = location
        return self.sensor_data[loc] if location in self.sensor_data else None

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
            
            tlist = []
            for (script, location) in self.device.scripts:
                
                thread = ExecuteScripts(self.device, location, \
                        neighbours, script)
                tlist.append(thread)
                thread.start()
            for thread in tlist:
                thread.join()
            
            
            self.device.barrier.wait()

from threading import Thread

class ExecuteScripts(Thread):
    
    def __init__(self, device, location, neighbours, script):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def run(self):
        
        script_data = []
        self.device.locks[self.location].acquire()
        
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
        self.device.locks[self.location].release()
