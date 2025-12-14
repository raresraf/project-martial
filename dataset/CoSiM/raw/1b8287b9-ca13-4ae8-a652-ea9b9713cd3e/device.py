


from threading import Event, Thread, Condition
from barrier import Barrier
from worker import Worker


class Device(object):
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.elocks = []
        self.barrier = Barrier(0)
        self.thread = DeviceThread(self)
        self.devices = []

    
    def __str__(self):
        
        return "Device %d" % self.device_id

    
    def create_conditions(self, condition_number):
        
        for _ in xrange(condition_number):
            condition_location = Condition()
            self.elocks.append(condition_location)


    def setup_devices(self, devices):
        
        length = len(devices)
        
        barrier = Barrier(length)

        
        if self.device_id == 0:
            condition_number = 0

            
            for device in devices:
                for location in device.sensor_data.keys():
                    if location > condition_number:
                        condition_number = location

            
            condition_number += 1

            
            for _ in xrange(condition_number):
                condition_location = Condition()


                self.elocks.append(condition_location)

            for device in devices:
                
                if barrier is not None:
                    device.barrier = barrier
                
                for j in xrange(condition_number):
                    device.elocks.append(self.elocks[j])
                
                device.thread.start()

    def assign_script(self, script, location):
        

        return self.script_received.set() if script is None \
            else self.scripts.append((script, location))


    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()



class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []


    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.script_received.wait()

            
            for (script, location) in self.device.scripts:
                worker_thread = Worker(self.device, script, location, neighbours)
                self.workers.append(worker_thread)

            length = len(self.workers)

            
            for i in range(length):
                self.workers[i].start()
            
            for i in range(length):
                self.workers[i].join()
            
            self.workers = []

            
            self.device.script_received.clear()
            self.device.barrier.wait()

from threading import Thread


class Worker(Thread):
    



    def __init__(self, device, script, location, neighbours):

        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def solve(self):
        
        
        self.device.elocks[self.location].acquire()
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

        
        self.device.elocks[self.location].release()

    def run(self):
        
        self.solve()
