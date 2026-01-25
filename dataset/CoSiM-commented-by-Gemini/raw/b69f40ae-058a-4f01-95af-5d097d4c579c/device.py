


from threading import Event, Thread, Lock, Condition


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
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.threads = []
        self.bariera = None
        self.lock = Lock()
        self.call_neigh = 1
        self.rupe = 0
        self.numara = 0
        self.neighbours = []
        self.numara_lock = Lock()
        self.call_neigh_lock = Lock()
        self.global_lock = None
        self.devices = []
        self.location_dict = {}

        i = 0
        while i < 8:
            self.threads.append(DeviceThread(self))
            i += 1



    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:
            self.bariera = ReusableBarrier(8*len(devices))
            self.global_lock = Lock()

            for dev in devices:
                dev.bariera = self.bariera
                dev.global_lock = self.global_lock
        self.devices = devices

        
        for i in xrange(8):
            self.threads[i].start()


    def assign_script(self, script, location):
        

        self.script_received.clear()

        if script is not None:
            self.scripts.append([script, location])
        else:
            self.timepoint_done.set()
        self.script_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        

        
        for i in xrange(8):
            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.lock = Lock()
        self.id_th = 0

    def run(self):

        
        self.device.numara_lock.acquire()
        self.id_th = self.device.numara
        self.device.numara += 1
        self.device.numara_lock.release()




        while True:
            
            self.device.call_neigh_lock.acquire()
            if self.device.call_neigh == 1:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                if self.device.neighbours is None:
                    self.device.rupe = 1
                    self.device.call_neigh_lock.release()
                    break
                self.device.call_neigh = 0
            self.device.call_neigh_lock.release()

            self.device.call_neigh_lock.acquire()
            if self.device.rupe == 1:
                self.device.call_neigh_lock.release()
                break
            self.device.call_neigh_lock.release()

            
            self.device.timepoint_done.wait()

            
            for i in xrange(self.id_th, len(self.device.scripts), 8):
                [script, location] = self.device.scripts[i]

                
                self.device.global_lock.acquire()
                if location not in self.device.location_dict:
                    self.device.location_dict[location] = Lock()
                    for j in xrange(len(self.device.devices)):
                        self.device.devices[j].location_dict[location] = \
                        self.device.location_dict[location]
                self.device.global_lock.release()

                self.device.location_dict[location].acquire()

                
                script_data = []

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
                        if device.get_data(location) is not None:
                            device.set_data(location, result)
                    
                    if self.device.get_data(location) is not None:
                        self.device.set_data(location, result)

                self.device.location_dict[location].release()

            
            self.device.bariera.wait()
            self.device.call_neigh = 1
            self.device.timepoint_done.clear()
            self.device.bariera.wait()
