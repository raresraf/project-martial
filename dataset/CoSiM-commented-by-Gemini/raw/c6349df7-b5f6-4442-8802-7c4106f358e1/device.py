


from threading import Event, Thread, Condition, Lock


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.scripts_done = Event()
        self.my_lock = Lock()

        
        
        self.locations = None
        self.barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        
        
        if self.device_id is 0:
            self.locations = {}
            self.barrier = ReusableBarrier(len(devices));
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass
                else:
                    self.locations[loc] = Lock()
        
        
        else:
            self.locations = devices[0].locations
            self.barrier = devices[0].get_barrier()
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass
                else:


                    self.locations[loc] = Lock()

        
        self.thread = DeviceThread(self, self.barrier, self.locations)
        self.thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

    def get_barrier(self):
        
        return self.barrier



class DeviceThread(Thread):
    

    def __init__(self, device, barrier, locations):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.locations = locations

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.scripts_done.wait()
            self.device.scripts_done.clear()
            
            

            workers = []
            for (script, location) in self.device.scripts:
                w = Worker(self.device, neighbours, script, location, self.locations)
                workers.append(w)
                w.start()

            
            for w in workers:
                w.join()

            
            self.barrier.wait()

class Worker(Thread):
    def __init__(self, device, neighbours, script, location, locations):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.locations = locations

    def run(self):
        
        self.locations[self.location].acquire()
        script_data = [] 
        
        for device in self.neighbours:
            device.my_lock.acquire()
            data = device.get_data(self.location)
            device.my_lock.release()
            if data is not None:
                script_data.append(data)
        
        
        self.device.my_lock.acquire()
        data = self.device.get_data(self.location)
        self.device.my_lock.release()
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            
            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()
            
        
        self.locations[self.location].release()



class ReusableBarrier():
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
 
    def wait(self):
        self.cond.acquire()                      
        self.count_threads -= 1;
        if self.count_threads == 0:
            self.cond.notify_all()               
            self.count_threads = self.num_threads    
        else:
            self.cond.wait();                    
        self.cond.release();                     