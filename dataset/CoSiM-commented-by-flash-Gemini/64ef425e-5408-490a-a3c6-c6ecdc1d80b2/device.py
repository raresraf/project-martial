


from threading import Event, Thread, Condition

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

class CoreThread(Thread):
    
    def __init__(self):
        Thread.__init__(self)
        self.threads = []
        self.results = []
    def append_script(self, script, location, data):
        
        self.threads.append((script, location, data))
    def run(self):
        
        self.results = [(script, location, script.run(data)) \
        for (script, location, data) in self.threads]



class Device(object):
    
    barrier = ReusableBarrierCond(0)
    barrier_set = False
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if not Device.barrier_set:
            Device.barrier = ReusableBarrierCond(len(devices))
            Device.barrier_set = True

        self.thread = DeviceThread(self)
        self.thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
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
            
            
            self.threads = [CoreThread() for i in range(8)]
            
            count = 0
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
                    self.threads[count].append_script(script, location, script_data)
                    count = (count+1) % 8

            
            for i in range(8):
                if self.threads[i].threads != []:
                    self.threads[i].start()
            
            for i in range(8):
                if self.threads[i].threads != []:
                    self.threads[i].join()

            
            Device.barrier.wait()

            for i in range(8):
                for (script, location, result) in self.threads[i].results:
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
            Device.barrier.wait()
