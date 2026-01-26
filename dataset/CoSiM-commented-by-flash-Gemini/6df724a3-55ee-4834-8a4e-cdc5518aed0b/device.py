


from threading import Event, Thread, Lock, Condition

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.bariera = ReusableBarrier(0)
        self.lacat_date = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            barria = ReusableBarrier(len(devices))
            for device in devices:
                device.bariera = barria

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()



class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_list = list()

    def run(self):
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            self.thread_list = list()
            for (script, location) in self.device.scripts:
                minithrd = MiniT(neighbours, self.device, location, script)
                self.thread_list.append(minithrd)

            for i in range(len(self.thread_list)):
                self.thread_list[i].start()

            for i in range(len(self.thread_list)):
                self.thread_list[i].join()

            self.device.timepoint_done.clear()
            self.device.bariera.wait()

class MiniT(Thread):
    
    def __init__(self, neighbours, device, location, script):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        script_data = []
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)



        if script_data != []:
            result = self.script.run(script_data)

            for device in self.neighbours:
                device.lacat_date.acquire()
                device.set_data(self.location, result)
                device.lacat_date.release()

            self.device.lacat_date.acquire()
            self.device.set_data(self.location, result)
            self.device.lacat_date.release()

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
