

import threading



from threading import Event, Thread, Lock, Semaphore, current_thread

class Device(object):
    
        
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.next_iteration = True  
        self.thread = DeviceThread(self)
        self.set_data_lock = Lock()
        self.step = 0


        self.all_devices = []
        self.all_devices_count = 0
        self.new_time = Event()
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.all_devices = devices
        self.all_devices_count = len(self.all_devices)

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            with self.set_data_lock:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

    def increment_step(self):
        with self.set_data_lock:
            self.step += 1
                        
class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
            while True: 
                self.device.new_time.set()
                
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    break
                
                self.device.timepoint_done.wait()
                run_thread = RunScripts(self.device, neighbours)
                run_thread.start()
                run_thread.join()

                self.device.increment_step()
                count = 0
                for d in self.device.all_devices:
                    if d.step == self.device.step:
                        count += 1
                        
                
                if count == self.device.all_devices_count:
                    for d in self.device.all_devices:
                        d.new_time.set()
                else:
                    self.device.new_time.wait()

class RunScripts(Thread):
    def __init__(self, device, neighbours):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours

    def run(self):
        self.device.new_time.clear()
        
        
        for (script, location) in self.device.scripts:
            script_data = []
            
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)
