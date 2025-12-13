


from threading import Event, Thread, Lock
from mybarrier import ReusableBarrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        
        self.reusable_barrier = None 
        self.devices = [] 
        self.location_lock = [] 
        self.set_lock = Lock() 
        self.get_lock = Lock() 
        self.ready = Event() 
                             
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        self.devices = devices
        if self.device_id == 0:
            reusable_barrier = ReusableBarrier(len(devices))
            for _ in range(100):
                self.location_lock.append(Lock())


            for dev in devices:
                dev.reusable_barrier = reusable_barrier
                dev.location_lock = self.location_lock
                dev.ready.set()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        
        with self.get_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        with self.set_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        self.device.ready.wait()
        while True:
            thread_list = []
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()
            self.device.script_received.clear()

            pos = 0

            for _ in self.device.scripts:
                thread_list.append(ExecutorThread(self.device, self.device.scripts, neighbours, pos))
                pos = pos + 1

            scripts_left = len(self.device.scripts)
            current_pos = 0

            if scripts_left < 8:
                for thread in thread_list:
                    thread.start()
                for thread in thread_list:
                    thread.join()
            else:
                while scripts_left >= 8:


                    for i in xrange(current_pos, current_pos + 8):
                        thread_list[i].start()
                    for i in xrange(current_pos, current_pos + 8):
                        thread_list[i].join()
                    current_pos = current_pos + 8
                    scripts_left = scripts_left - 8


                for i in xrange(current_pos, current_pos + scripts_left):
                    thread_list[i].start()
                for i in xrange(current_pos, current_pos + scripts_left):
                    thread_list[i].join()
            self.device.reusable_barrier.wait()

class ExecutorThread(Thread):
    
    def __init__(self, device, scripts, neighbours, pos):
        
        Thread.__init__(self, name="Executor Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours
        self.pos = pos

    def run(self):
        
        (script, location) = self.scripts[self.pos]
        self.device.location_lock[location].acquire()
        script_data = []
        for dev in self.neighbours:
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        if script_data != []:
            result = script.run(script_data)
            for dev in self.neighbours:
                dev.set_data(location, result)
                self.device.set_data(location, result)
        self.device.location_lock[location].release()
