

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

        self.timepoint_done = Event()
        self.script_received = Event()
        self.barrier = None
        self.location_lock = None
        self.lock = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == 0:
            self.lock = Lock()
            self.barrier = ReusableBarrierSem(len(devices))
            self.location_lock = {}
            for device in devices:
                device.location_lock = self.location_lock
                for location in device.sensor_data:
                    self.location_lock[location] = Lock()
                    if device.device_id != 0:
                        device.barrier = self.barrier
                        device.lock = self.lock


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()
            

    def get_data(self, location):
        
        with self.lock:
            res = self.sensor_data[location] if location in self.sensor_data else None
        return res

    def set_data(self, location, data):
        
        with self.lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

    def run_script(self, script, location, neighbours):
        
        self.location_lock[location].acquire()
        script_data = []
        
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
            
        data = self.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            


            result = script.run(script_data)
            
            for device in neighbours:
                device.set_data(location, result)
            self.set_data(location, result)

        
        self.location_lock[location].release()
        

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

            
            thread_list = []

            
            for (script, location) in self.device.scripts:
                
                if len(thread_list) < 8:
                    
                    t = Thread(target=self.device.run_script, args=(script, location, neighbours))
                    t.start()
                    thread_list.append(t)
                else:
                    
                    out_thread = thread_list.pop(0)
                    out_thread.join()

            
            for thread in thread_list:
                thread.join()
                
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
            
