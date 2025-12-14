


from threading import Event, Thread, Lock, Condition


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
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.my_lock = Lock()


        self.local_lock = Lock()
        self.setup_device = Event()
        self.device_barrier = None
        self.local_barrier = ReusableBarrierCond(8)
        self.location_lock = {}
        self.neighbours = []
        self.threads = []
        for i in range(8):
            self.threads.append(DeviceThread(self, i))
        for thread in self.threads:
            thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            my_barrier = ReusableBarrierCond(len(devices)*8)
            my_location_lock = {}
            for device in devices:
                device.device_barrier = my_barrier
                device.location_lock = my_location_lock
                device.setup_device.set()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            if location not in self.location_lock:
                self.location_lock[location] = Lock()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        val = self.sensor_data[location] if location in self.sensor_data else None
        return val

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
        
        self.device.setup_device.wait()

        while True:
            self.device.device_barrier.wait()
            index = self.thread_id

            
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            self.device.local_barrier.wait()

            if self.device.neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            while index < len(self.device.scripts):
                (script, location) = self.device.scripts[index]
                index += 8
                script_data = []

                self.device.location_lock[location].acquire()

                
                for device in self.device.neighbours:
                    data = device.get_data(location)

                    if data is not None:
                        script_data.append(data)

                
                data = self.device.get_data(location)

                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in self.device.neighbours:
                        device.set_data(location, result)

                    
                    self.device.set_data(location, result)

                self.device.location_lock[location].release()

            self.device.device_barrier.wait()
            if self.thread_id == 0:
                self.device.timepoint_done.clear()
