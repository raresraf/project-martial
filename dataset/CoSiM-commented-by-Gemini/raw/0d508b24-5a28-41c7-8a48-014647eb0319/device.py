


from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []


        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.lock = None
        self.lock2 = None
        self.locks = []
        self.reusable_barrier = None
        self.script_threads = []
        self.shared_obj = None

    def __str__(self):
        
        return "Device {}".format(self.device_id)

    def setup_devices(self, devices):
        

        if self == devices[0]:
            
            
            
            lock = Lock()
            lock2 = Lock()
            barrier = ReusableBarrier(len(devices))
            shared_obj = SharedObjects()

            
            for device in devices:
                device.lock = lock
                device.reusable_barrier = barrier
                device.lock2 = lock2
                device.shared_obj = shared_obj

        if self == devices[-1]:
            
            
            for device in devices:
                device.thread.start()

    def assign_script(self, script, location):
        

        
        
        
        

        self.lock2.acquire()
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
            self.script_received.set()
        self.lock2.release()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        self.lock2.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock2.release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        position = 0

        while True:
            
            
            self.device.reusable_barrier.wait()

            
            self.device.lock.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.lock.release()

            if neighbours is None:
                break



            self.device.reusable_barrier.wait()

            
            
            self.device.script_received.wait()
            self.device.script_received.clear()

            

            
            
            
            for (script, location) in self.device.scripts:
                self.device.lock.acquire()
                location_ok = True
                for (loc, _) in self.device.shared_obj.locations_lock:
                    if loc == location:
                        location_ok = False
                        break
                if location_ok:
                    self.device.shared_obj.locations_lock.append((location,
                                                                  Lock()))
                self.device.lock.release()
                
                
                
                
                script_thread = ScriptThread(self.device, script,
                                             location, neighbours, position)
                self.device.script_threads.append(script_thread)
                position += 1
                if position == 8:
                    position = 0
                    for script_thread in self.device.script_threads:
                        script_thread.start()
                    for script_thread in self.device.script_threads:
                        script_thread.join()
                    self.device.script_threads = []

            
            for script_thread in self.device.script_threads:
                script_thread.start()

            for script_thread in self.device.script_threads:
                script_thread.join()

            self.device.script_threads = []

            self.device.reusable_barrier.wait()
            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()


class SharedObjects(object):
    

    def __init__(self):
        self.locations_lock = []


class ScriptThread(Thread):
    

    def __init__(self, device, script, location, neighbours, position):
        
        name = "Device Thread {}, Script {}".format(device.device_id, script)
        Thread.__init__(self, name=name)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.position = position

    def run(self):
        script_data = []
        lock = None
        for (location, a_lock) in self.device.shared_obj.locations_lock:
            if location == self.location:
                lock = a_lock
                break

        lock.acquire()

        
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

        lock.release()
