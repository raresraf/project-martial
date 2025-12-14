


from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

class Device(object):
    


    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.bariera = None
        self.timepoint_done = Event()
        self.threads = []
        self.nr_threads = 8
        self.locks = {}
        self.bariera_interioara = ReusableBarrierCond(self.nr_threads)


        for index in xrange(0, self.nr_threads):
            thread = DeviceThread(self, index)
            self.threads.append(thread)

        for thread in self.threads:
            thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            
            
            self.bariera = ReusableBarrierCond(len(devices))
            for dev in devices:
                dev.bariera = self.bariera
            
            max_location = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > max_location:
                        max_location = location
            
            
            for location in xrange(0, max_location + 1):
                self.locks[location] = Lock()
            for device in devices:
                device.locks = self.locks


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
        
        for thread in self.threads:
            thread.join()





class DeviceThread(Thread):
    

    def __init__(self, device, index):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.index = index

    def run(self):

        while True:
            
            self.device.bariera_interioara.wait()
            if self.index == 0:
                
                self.device.neighbours = self.device.supervisor.get_neighbours()

            
            self.device.bariera_interioara.wait()
            if self.device.neighbours is None:
                break

            if self.index == 0:
                self.device.timepoint_done.wait()
                self.device.bariera.wait()
            
            self.device.bariera_interioara.wait()


            
            for index in xrange(0, len(self.device.scripts)):
                
                if self.index == index % self.device.nr_threads:
                    (script, location) = self.device.scripts[index]
                    self.device.locks[location].acquire()
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
                            device.set_data(location, result)
                        
                        self.device.set_data(location, result)
                    self.device.locks[location].release()
            if self.index == 0:
                self.device.timepoint_done.clear()

