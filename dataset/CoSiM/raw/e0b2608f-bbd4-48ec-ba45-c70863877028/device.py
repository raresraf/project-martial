



from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        
        
        
        self.sensor_data_lock = Lock()
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        
        self.setup_device = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        
        self.barrier = None
        self.devices = []
        self.locations = []


    def __str__(self):
        
        return "Device %d" % self.device_id


    def set_locks(self):
        
        i = 0
        while i < 100:
            self.locations.append(Lock())
            i = i + 1


    def setup_devices(self, devices):
        
        
        self.devices = devices
        nr_devices = len(devices)
        
        barrier_setup = ReusableBarrierSem(nr_devices)

        if self.device_id == 0:
            
            
            self.set_locks()
            for device in devices:
                
                device.locations = self.locations
                
                if device.barrier is None:
                    device.barrier = barrier_setup
                
                device.setup_device.set()


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        
        
        with self.sensor_data_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        
        
        with self.sensor_data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()



class MyThread(Thread):
    

    def __init__(self, device, location, neighbours, script):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.location = location
        self.script = script

    def run(self):
        
        
        self.device.locations[self.location].acquire()
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
        self.device.locations[self.location].release()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        self.device.setup_device.wait()
        
        while 1:
            threads = []
            current_script_id = 0
            start = 0
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()
            scripts_received = len(self.device.scripts)
            
            
            while current_script_id < scripts_received:
                (script, location) = self.device.scripts[current_script_id]
                thread = MyThread(self.device, location, neighbours, script)
                threads.append(thread)
                current_script_id = current_script_id + 1

            
            
            
            if scripts_received < 8:
                for threadd in threads:
                    threadd.start()
                for threadd in threads:
                    threadd.join()
            else:
                while 1:
                    if scripts_received >= 8:


                        stop = start+8
                        for i in range(start, stop):
                            threads[i].start()
                        for i in range(start, stop):
                            threads[i].join()
                        start = start+8
                        scripts_received = scripts_received - 8
                    else:


                        stop = start+scripts_received
                        for i in range(start, stop):
                            threads[i].start()
                        for i in range(start, stop):
                            threads[i].join()
                        break
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
