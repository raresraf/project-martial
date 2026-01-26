


from threading import Condition, Lock, Event, Thread

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
    



    barrier = None      
    unique = []   

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []   
        self.timepoint_done = Event()   

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if Device.barrier == None:  
            Device.barrier = ReusableBarrierCond(len(devices))

        
        
        if len(Device.unique) != \
        self.supervisor.supervisor.testcase.num_locations:
            for _ in \
            range(self.supervisor.supervisor.testcase.num_locations):
                Device.unique.append(Lock())

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data \
                                          else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class ScriptThread(Thread):
    

    def __init__(self, device, scripts, neighbours):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        
        for (script, location) in self.scripts:
            Device.unique[location].acquire()

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

            Device.unique[location].release()

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

            
            
            threads = []
            divide_scripts = [[] for _ in range(len(self.device.scripts))]

            for i in range(len(self.device.scripts)):
                divide_scripts[i].append(self.device.scripts[i])

            for i in range(len(divide_scripts)):
                threads.append(ScriptThread(self.device, divide_scripts[i], \
                                            neighbours))

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            Device.barrier.wait() 
            self.device.timepoint_done.clear()
