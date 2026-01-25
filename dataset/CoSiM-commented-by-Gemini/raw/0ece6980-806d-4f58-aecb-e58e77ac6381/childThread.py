
from Queue import Queue
from threading import Thread


class Child(object):
    

    def __init__(self):
        
        self.que = Queue(8)
        self.device = None
        self.threads = []

        for _ in xrange(8):
            new_thread = Thread(target=self.run)
            self.threads.append(new_thread)

        for thread in self.threads:
            thread.start()

    def set_device(self, device):
        
        self.device = device

    def submit(self, neighbours, script, location):
        
        self.que.put((neighbours, script, location))

    def wait(self):
        
        self.que.join()

    def end_threads(self):
        
        self.wait()
        
        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)

        for thread in self.threads:
            thread.join()

    def run(self):
        



        while True:

            (neighbours, script, location) = self.que.get()

            if neighbours is None and script is None:
                self.que.task_done()
                return

            script_data = []
            
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    
                    if data is not None:
                        script_data.append(data)

            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            
            if script_data != []:

                rezult = script.run(script_data)

                for device in neighbours:


                    if device.device_id != self.device.device_id:
                        device.set_data(location, rezult)

                self.device.set_data(location, rezult)

            self.que.task_done()

from threading import Event, Lock, Thread
from myBarrier import MyBarrier
from childThread import Child


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()


        self.scripts = []
        self.timepoint_done = Event()
        
        self.barrier = None
        
        self.data_lock = {location : Lock() for location in sensor_data}
        
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:
            self.barrier = MyBarrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            self.data_lock[location].acquire()
            return self.sensor_data[location]
        else:
            return None
        

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_lock[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.child_threads = Child()

    def run(self):

        self.child_threads.set_device(self.device)
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            while True:

                
                if not self.device.script_received.isSet():
                    self.device.timepoint_done.wait()
                    
                    if not self.device.script_received.isSet():
                        
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break
                else:
                    
                    self.device.script_received.clear()
                    
                    for (script, location) in self.device.scripts:
                        self.child_threads.submit(neighbours, script, location)

            
            self.child_threads.wait()

            
            self.device.barrier.wait()

        self.child_threads.end_threads()
