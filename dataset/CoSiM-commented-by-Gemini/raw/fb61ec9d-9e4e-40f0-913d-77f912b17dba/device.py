


from threading import Event, Thread, Lock, Semaphore
from ReusableBarrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []

        
        self.lock_setter = Lock()
        self.lock_getter = Lock()
        self.lock_assign = Lock()

        
        self.barrier = None
        self.location_lock = {}

        
        self.semaphore = Semaphore(8)


        self.thread = DeviceThread(self)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))

            
            for device in devices[:]:
                for loc in device.sensor_data.keys():
                    if loc not in self.location_lock:
                        self.location_lock[loc] = Lock()

            
            for device in devices[:]:
                device.barrier = self.barrier
                device.location_lock = self.location_lock
                
                device.thread.start()


    def assign_script(self, script, location):
        

        with self.lock_assign:

            if script is not None:
                self.scripts.append((script, location))
            else:
                self.script_received.set()

    def get_data(self, location):
        

        with self.lock_getter:

            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        

        with self.lock_setter:

            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class ScriptThread(Thread):
    

    def __init__(self, device_thread, script, location, neighbours):
        
        Thread.__init__(self)
        self.script = script
        self.device_thread = device_thread


        self.location = location
        self.neighbours = neighbours

    def run(self):
        
        self.device_thread.device.location_lock[self.location].acquire()

        self.device_thread.device.semaphore.acquire()

        script_data = []
        
        for device in self.neighbours:


            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device_thread.device.set_data(self.location, result)

        self.device_thread.device.semaphore.release()
        self.device_thread.device.location_lock[self.location].release()




class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break



            self.device.script_received.wait()
            script_threads = []

            
            for (script, location) in self.device.scripts:
                
                thread = ScriptThread(self, script, location, neighbours)
                script_threads.append(thread)
                thread.start()

            for thread in script_threads:
                thread.join()

            self.device.script_received.clear()

            
            self.device.barrier.wait()
