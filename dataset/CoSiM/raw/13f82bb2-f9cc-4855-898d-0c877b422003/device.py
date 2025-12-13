


from threading import Event, Thread, Lock
import my_barrier
import my_thread

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

        self.devices = []
        self.threads = []
        self.barrier = None
        self.lock = [None] * 50

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        for device in devices:
            self.devices.append(device)
        
        self.barrier = my_barrier.ReusableBarrierCond(len(devices))
        for device in devices:
            device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
            
            for device in self.devices:
                if self.lock[location] is None and device.lock[location] is not None:
                    self.lock[location] = device.lock[location]
            if self.lock[location] is None:
                self.lock[location] = Lock()
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

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break



            self.device.timepoint_done.wait()
            
            for (script, location) in self.device.scripts:
            	
                thread = my_thread.RunScripts(self.device, neighbours, location, script)
                self.device.threads.append(thread)
            for i in xrange(len(self.device.threads)):
                self.device.threads[i].start()
            for i in xrange(len(self.device.threads)):


                self.device.threads[i].join()

            self.device.threads = []
            self.device.barrier.wait()
            self.device.timepoint_done.clear()


from threading import Thread

class RunScripts(Thread):
    
    def __init__(self, device, neighbours, location, script):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.location = location
        self.script = script

    def run(self):
        self.device.lock[self.location].acquire()
        self.device.script_received.wait()
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
                device.set_data(self.location, result)
    	    	
            self.device.set_data(self.location, result)
        self.device.lock[self.location].release()

