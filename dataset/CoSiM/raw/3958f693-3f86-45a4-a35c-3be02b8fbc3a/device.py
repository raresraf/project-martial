


from threading import Event, Thread, Lock, Condition
from reusable_barrier import ReusableBarrier

NUM_THREADS = 8

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.supervisor = supervisor

        
        
        self.ready_to_start = Event()

        
        
        self.data_lock = Lock()
        self.sensor_data = sensor_data

        
        self.location_busy = {location: False for location in self.sensor_data}
        self.location_busy_lock = Lock()

        
        self.scripts = []
        self.scripts_assigned = False
        self.scripts_enabled = False
        self.scripts_started_idx = 0
        self.scripts_done_idx = 0

        
        
        self.scripts_lock = Lock()
        self.scripts_condition = Condition(self.scripts_lock)
        self.scripts_done_condition = Condition(self.scripts_lock)

        
        
        self.thread_running = True
        self.thread = DeviceThread(self)
        self.worker_threads = [ScriptWorker(self, i) for i in range(NUM_THREADS)]

        
        for thread in self.worker_threads:
            thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == 0:
            timestep_barrier = ReusableBarrier(len(devices))
            location_conditions = {}

            
            for device in devices:
                for location in device.sensor_data:
                    if not location in location_conditions:
                        location_conditions[location] = Condition()

            for device in devices:
                device.location_conditions = location_conditions
                device.timestep_barrier = timestep_barrier

            for device in devices:
                device.ready_to_start.set()

        self.thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts_lock.acquire()
            self.scripts.append((script, location))
            self.scripts_condition.notify_all()
            self.scripts_lock.release()
        else:
            self.scripts_lock.acquire()
            self.scripts_assigned = True
            self.scripts_done_condition.notify_all()
            self.scripts_lock.release()

    def is_busy(self, location):
        self.location_busy_lock.acquire()
        ret = location in self.location_busy and self.location_busy[location]
        self.location_busy_lock.release()
        return ret

    def set_busy(self, location, value):
        self.location_busy_lock.acquire()
        self.location_busy[location] = value
        self.location_busy_lock.release()

    def has_data(self, location):
        self.data_lock.acquire()
        ret = location in self.sensor_data
        self.data_lock.release()
        return ret

    def get_data(self, location):
        
        self.data_lock.acquire()
        ret = self.sensor_data[location] if location in self.sensor_data else None
        self.data_lock.release()
        return ret

    def set_data(self, location, data):
        
        self.data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.data_lock.release()

    def shutdown(self):
        
        self.thread.join()
        for thread in self.worker_threads:
            thread.join()

class ScriptWorker(Thread):
    def __init__(self, device, index):
        Thread.__init__(self, name="Worker thread %d for device %d" % (index, device.device_id))
        self.device = device
        self.lock = device.scripts_lock
        self.done_condition = device.scripts_done_condition
        self.condition = device.scripts_condition

    def run(self):
        
        self.lock.acquire()

        while self.device.thread_running:
            script = None

            
            if not self.device.scripts_enabled or \
                   self.device.scripts_started_idx >= len(self.device.scripts):
                self.condition.wait()
                continue

            
            script = self.device.scripts[self.device.scripts_started_idx]
            self.device.scripts_started_idx = self.device.scripts_started_idx + 1
            self.condition.notify_all()

            
            self.lock.release()
            self.run_script(script[0], script[1])
            self.lock.acquire()

            
            self.device.scripts_done_idx = self.device.scripts_done_idx + 1
            self.done_condition.notify_all()

        
        self.lock.release()


    def run_script(self, script, location):
        
        
        self.device.location_conditions[location].acquire()

        
        script_devices = []
        for device in self.device.neighbours:
            if device.has_data(location):
                script_devices.append(device)
        if self.device.has_data(location):
            script_devices.append(self.device)

        
        if len(script_devices) == 0:
            self.device.location_conditions[location].release()
            return

        
        while True:
            free = True
            for device in script_devices:
                if device.is_busy(location):
                    free = False
                    break
            if free:
                break
            self.device.location_conditions[location].wait()

        
        script_data = []
        for device in script_devices:
            device.set_busy(location, True)
            script_data.append(device.get_data(location))
        self.device.location_conditions[location].notify_all()

        
        self.device.location_conditions[location].release()
        result = script.run(script_data)
        self.device.location_conditions[location].acquire()

        
        for device in script_devices:
            device.set_data(location, result)
            device.set_busy(location, False)
        self.device.location_conditions[location].notify_all()

        
        self.device.location_conditions[location].release()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        self.device.ready_to_start.wait()

        while True:
            
            self.device.timestep_barrier.wait()

            


            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                break

            
            self.device.scripts_lock.acquire()
            self.device.scripts_started_idx = 0
            self.device.scripts_done_idx = 0
            self.device.scripts_enabled = True
            self.device.scripts_condition.notify_all()
            self.device.scripts_lock.release()

            
            self.device.scripts_lock.acquire()
            while not self.device.scripts_assigned or \
                  self.device.scripts_done_idx < len(self.device.scripts):
                self.device.scripts_done_condition.wait()
            self.device.scripts_enabled = False
            self.device.scripts_lock.release()

        
        self.device.scripts_lock.acquire()
        self.device.thread_running = False
        self.device.scripts_condition.notify_all()
        self.device.scripts_lock.release()
