


from threading import Event, Thread, Condition, Lock

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
        
        self.root_device = 0
        self.init_lock = Lock()
        self.finalize_lock = Lock()
        self.max_threads = 8
        self.device_barrier = ReusableBarrierCond(self.max_threads)

        self.neighbours = None
        self.barrier = None
        self.locks = None
        self.dict_lock = None

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.threads = [DeviceThread(self, i) for i in range(self.max_threads)]

        for thread in self.threads:
            thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.device_id == self.root_device:
            self.locks = {}
            self.dict_lock = Lock()
            self.barrier = ReusableBarrierCond(self.max_threads * len(devices))

            
            for device in devices:
                if device.device_id != self.root_device:
                    device.dict_lock = self.dict_lock
                    device.locks = self.locks
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):


        while True:
            
            self.device.init_lock.acquire()

            if self.device.neighbours is None:
                self.device.neighbours = self.device.supervisor.get_neighbours()
            neighbours = self.device.neighbours

            self.device.init_lock.release()

            


            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            
            for i in range(self.thread_id, len(self.device.scripts), self.device.max_threads):
                (script, location) = self.device.scripts[i]

                
                self.device.dict_lock.acquire()
                if location not in self.device.locks:
                    self.device.locks[location] = Lock()
                self.device.dict_lock.release()

                
                self.device.locks[location].acquire()

                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    
                    for device in neighbours:
                        device.set_data(location, result)
                    


                    self.device.set_data(location, result)

                self.device.locks[location].release()

            
            self.device.device_barrier.wait()

            
            self.device.finalize_lock.acquire()
            if self.device.neighbours is not None:
                self.device.neighbours = None
                self.device.timepoint_done.clear()
            self.device.finalize_lock.release()

            
            self.device.barrier.wait()
