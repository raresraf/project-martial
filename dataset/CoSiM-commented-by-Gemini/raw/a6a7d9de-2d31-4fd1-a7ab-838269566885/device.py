




from threading import Event, Thread, BoundedSemaphore, Lock
from cond_barrier import ReusableBarrier


class Device(object):
    
    barrier = None
    barrier_event = Event()

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        
        self.scripts = []
        self.scripts_semaphore = BoundedSemaphore(8)
        self.scripts_lock = Lock()

        
        self.script_received = Event()
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        


        if Device.barrier is None and self.device_id == 0:
            Device.barrier = ReusableBarrier(len(devices))
            Device.barrier_event.set()

    def assign_script(self, script, location):
        
        
        with self.scripts_lock:
            self.script_received.set()
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
        
        self.thread.join()


class DeviceThread(Thread):
    
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        Device.barrier_event.wait()
        while True:
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break

            
            script_index = 0
            script_threads = []
            while True:
                self.device.scripts_lock.acquire()
                if script_index < len(self.device.scripts):
                    
                    self.device.scripts_lock.release()
                    self.device.scripts_semaphore.acquire()

                    script_threads.append(ScriptThread(self.device, self.device.scripts[script_index][0],
                                                       self.device.scripts[script_index][1], neighbours))
                    script_threads[-1].start()

                    script_index += 1
                else:
                    
                    if self.device.timepoint_done.is_set() and script_index == len(self.device.scripts):
                        self.device.timepoint_done.clear()
                        self.device.scripts_lock.release()
                        break
                    else:
                        
                        self.device.scripts_lock.release()
                        self.device.script_received.wait()
                        self.device.scripts_lock.acquire()
                        self.device.script_received.clear()
                        self.device.scripts_lock.release()

            
            for script_thread in script_threads:
                script_thread.join()

            
            Device.barrier.wait()


class ScriptThread(Thread):
    
    locations_locks = {}

    def __init__(self, device, script, location, neighbours):
        


        Thread.__init__(self)
        self.location = location
        self.script = script
        self.device = device
        self.neighbours = neighbours

        if location not in ScriptThread.locations_locks:
            ScriptThread.locations_locks[location] = Lock()

    def run(self):
        
        with ScriptThread.locations_locks[self.location]:
            
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

        
        self.device.scripts_semaphore.release()
