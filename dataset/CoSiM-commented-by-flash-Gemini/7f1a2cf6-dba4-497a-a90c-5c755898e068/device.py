


from threading import Event, Thread, Lock
from multiprocessing import cpu_count
from barrier import *


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = None
        self.thread = DeviceThread(self)
        self.neighbours = []
        self.locks = None
        self.max_minions = max(8, cpu_count())

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            
            barrier = ReusableBarrierCond(len(devices))

            
            locks = {}

            
            for dev in devices:
                for pair in dev.sensor_data:
                    if not pair in locks:
                        locks[pair] = Lock()

            
            for dev in devices:
                dev.timepoint_done = barrier
                dev.locks = locks
                dev.thread.start()



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
        
        minions = []
        while True:
            


            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                break

            
            self.device.script_received.wait()
            self.device.script_received.clear()

            
            tasks = {}
            for i in xrange(self.device.max_minions):
                tasks[i] = []

            
            for i in xrange(len(self.device.scripts)):
                tasks[i % self.device.max_minions].append(self.device.scripts[i])

            
            for i in xrange(self.device.max_minions):
                if len(tasks[i]) > 0:
                    minions.append(Thread(target=run_task, args=(self.device, tasks[i])))

            for minion in minions:
                minion.start()

            
            for minion in minions:
                minion.join()

            
            while len(minions) > 0:
                minions.remove(minions[0])

            
            self.device.timepoint_done.wait()

def run_task(device, tasks):
    
    for task in tasks:
        (script, location) = task
        script_data = []

        
        device.locks[location].acquire()

        
        for dev in device.neighbours:
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)

        
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)

        
        if script_data != []:
            
            result = script.run(script_data)

            
            for dev in device.neighbours:
                dev.set_data(location, result)

            
            device.set_data(location, result)

        
        device.locks[location].release()
