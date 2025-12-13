


from threading import Event, Thread, Lock, RLock
from Queue import Queue


class Device(object):
    
    no_cores = 8

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.sensor_data_locks = {}
        self.supervisor = supervisor
        self.devices_other = []

        self.scripts = []
        self.script_queue = Queue()
        self.scripts_lock = Lock()
        self.virt_socket = Queue()

        self.start_lock = Lock()
        self.start_is_at = True
        self.end_event = Event()

        self.neighbours = []
        self.counter = 1

        self.threads = []
        for _ in range(Device.no_cores):
            self.threads.append(DeviceThread(self))
        self.active_threads = 1
        self.threads[0].start()

    def __start_thread(self):
        
        
        if self.active_threads >= Device.no_cores:
            return

        no_thr = len(self.scripts)
        if no_thr > self.active_threads:
            self.threads[self.active_threads].start()
            self.active_threads += 1

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.devices_other = devices
        self.devices_other.remove(self)
        for loc in self.sensor_data:
            self.sensor_data_locks[loc] = RLock()

    def sync_send(self):
        
        self.virt_socket.put(None)

    def sync_devices(self):
        
        for dev in self.devices_other:
            dev.sync_send()
        for _ in self.devices_other:
            _ = self.virt_socket.get()

    def assign_script(self, script, location):
        
        if script is not None:
            with self.scripts_lock:
                self.script_queue.put((script, location))
                self.scripts.append((script, location))
                self.__start_thread()
        else:
            with self.scripts_lock:
                for _ in range(self.active_threads):
                    self.script_queue.put(None)

    def get_data(self, location):
        
        if location in self.sensor_data:
            with self.sensor_data_locks[location]:
                return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            with self.sensor_data_locks[location]:
                self.sensor_data[location] = data

    def get_data_lock(self, location):
        
        if location in self.sensor_data:
            self.sensor_data_locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data_unlock(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.sensor_data_locks[location].release()

    def timepoint_init(self):
        
        with self.scripts_lock:
            for script in self.scripts:
                self.script_queue.put(script)
        self.neighbours = self.supervisor.get_neighbours()
        if self.neighbours is not None:
            self.neighbours = list(self.neighbours)
            self.neighbours.append(self)
            
            self.neighbours.sort(key=lambda x: x.device_id)

    def shutdown(self):
        
        for thr in self.threads:
            if thr.isAlive():
                thr.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        while True:
            
            with self.device.start_lock:
                if self.device.start_is_at:
                    self.device.start_is_at = False
                    self.device.timepoint_init()
                    self.device.end_event.clear()

            neighbours = self.device.neighbours
            if neighbours is None:
                break

            
            while True:
                
                pair = self.device.script_queue.get()
                if pair is None:
                    self.device.script_queue.task_done()
                    break
                script = pair[0]
                location = pair[1]

                script_data = []
                
                for device in neighbours:
                    data = device.get_data_lock(location)
                    if data is not None:
                        script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in neighbours:
                        device.set_data_unlock(location, result)

                self.device.script_queue.task_done()

            
            self.device.script_queue.join()

            
            with self.device.start_lock:
                if not self.device.start_is_at:
                    self.device.start_is_at = True
                    self.device.sync_devices()
                    
                    self.device.counter = self.device.active_threads - 1
                else:
                    self.device.counter = self.device.counter - 1
                if self.device.counter == 0:
                    self.device.end_event.set()
            self.device.end_event.wait()
