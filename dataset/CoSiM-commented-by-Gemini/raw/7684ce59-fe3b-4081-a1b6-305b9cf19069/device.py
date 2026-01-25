


from threading import Event, Thread, Condition, Lock

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.scripts_dict = {}
        self.locations_locks = {}
        self.timepoint_done = None
        self.neighbours = None
        self.thread = DeviceThread(self)
        self.thread.start()


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        
        nr_devices = len(devices)
        if self.timepoint_done is None:
            self.timepoint_done = ReusableBarrierCond(nr_devices)
            for device in devices:
                if device.timepoint_done is None and device != self:
                    device.timepoint_done = self.timepoint_done

        
        
        
        for location in self.sensor_data.keys():
            if location not in self.locations_locks:
                self.locations_locks[location] = Lock()
                for device in devices:
                    if location not in device.locations_locks and \
                        device != self:
                        device.locations_locks[location] = \
                            self.locations_locks[location]



    def assign_script(self, script, location):
        
        if script is not None:
            
            if location in self.scripts_dict:
                self.scripts_dict[location].append(script)
            else:
                self.scripts_dict[location] = []
                self.scripts_dict[location].append(script)
        else:
            
            self.scripts_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] \
            if location in self.sensor_data else None

    def set_data(self, location, data):
        

        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                break

            
            self.device.scripts_received.wait()

            
            
            threads = []
            for location in self.device.scripts_dict.keys():
                thread = DeviceWorkerThread(self.device, location)
                thread.start()
                threads.append(thread)

            
            for thread in threads:
                thread.join()

            
            
            self.device.scripts_received.clear()

            
            self.device.timepoint_done.wait()

class DeviceWorkerThread(Thread):
    

    def __init__(self, device, location):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.location = location

    def run(self):
        
        for script in self.device.scripts_dict[self.location]:

            
            self.device.locations_locks[self.location].acquire()

            script_data = []
            
            for device in self.device.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)
                
                for device in self.device.neighbours:
                    device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)

            
            self.device.locations_locks[self.location].release()


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

