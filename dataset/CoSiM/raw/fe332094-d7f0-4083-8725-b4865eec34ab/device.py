


from threading import Thread, Condition, Lock

class ReusableBarrier(object):
    
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
        self.scripts_received = ReusableBarrier(9)
        self.scripts = {} 
        self.devices = None
        self.timepoint_done = None
        self.semafor = {}
        self.thread_list = []
        self.neighbours_barrier = ReusableBarrier(8)
        self.contor = 0

        for i in range(8):
            self.thread_list.append(DeviceThread(self, i))
            self.scripts.update({i:[]})

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        

        if self.device_id == 0:
            self.devices = devices
            
            self.timepoint_done = ReusableBarrier(8 * len(self.devices))
            
            for device in self.devices:
                
                for location in device.sensor_data:
                    if location not in self.semafor:
                        self.semafor.update({location: Lock()})
                if device.device_id != 0:
                    
                    device.initialize_device(self.timepoint_done, self.semafor, self.devices)
            
            for thread in self.thread_list:
                thread.start()

    def initialize_device(self, timepoint_done, semafor, devices):
        
        self.timepoint_done = timepoint_done
        self.semafor = semafor
        self.devices = devices
        
        for thread in self.thread_list:
            thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            
            self.scripts[self.contor%8].append((script, location))
            self.contor += 1
        else:
            
            self.scripts_received.wait()

    def get_data(self, location):
        

        value = None
        if location in self.sensor_data:
            value = self.sensor_data[location]
        return value

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id
        self.neighbours = None

    def initialize_neighbours(self, neighbours):
        
        self.neighbours = neighbours

    def run(self):
        
        while True:

            
            if self.thread_id == 0:
                self.neighbours = self.device.supervisor.get_neighbours()
                for thread in self.device.thread_list:
                    if thread.thread_id != 0:
                        
                        thread.initialize_neighbours(self.neighbours)

            self.device.neighbours_barrier.wait()

            if self.neighbours is None:
                self.device.timepoint_done.wait()
                break

            
            self.device.scripts_received.wait()

            
            for (script, location) in self.device.scripts[self.thread_id]:
                
                self.device.semafor[location].acquire()
                script_data = []
                
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                    if len(script_data) == 1:
                        self.device.semafor[location].release()
                        continue

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in self.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                
                self.device.semafor[location].release()

            
            
            self.device.timepoint_done.wait()
