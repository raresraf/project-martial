


from threading import Event, Thread, Lock, Condition, RLock
from Queue import Queue
from barrier import ReusableBarrierCond


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        

        self.sync_barrier = None
        self.acquire_stage_lock = None
        self.device_init_event = Event()

        self.location_data_lock = {location:RLock() for location in sensor_data}

        self.device_id = device_id
        self.supervisor = supervisor
        self.sensor_data = sensor_data

        self.timepoint_ended = False
        self.script_condition = Condition()
        self.scripts = []

        self.thread = DeviceThread(0, self)
        self.worker_pool = [DeviceWorker(i, self) for i in xrange(1, 9)]

        self.neighbours = []
        self.work_queue = Queue()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            
            self.acquire_stage_lock = Lock()
            self.sync_barrier = ReusableBarrierCond(len(devices))
        else:
            
            for device in devices:
                if device.device_id == 0:
                    device.device_init_event.wait()
                    self.sync_barrier = device.sync_barrier
                    self.acquire_stage_lock = device.acquire_stage_lock
        self.device_init_event.set()
        self.thread.start()

    def assign_script(self, script, location):
        
        with self.script_condition:
            while self.timepoint_ended:
                self.script_condition.wait()
            if script is not None:
                self.scripts.append((script, location))
            else:
                self.timepoint_ended = True
                self.script_condition.notify_all()

    def get_data(self, location):
        
        if location in self.sensor_data:
            with self.location_data_lock[location]:
                return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            with self.location_data_lock[location]:
                self.sensor_data[location] = data

    def acquire_location(self, location):
        
        if location in self.location_data_lock:
            self.location_data_lock[location].acquire()
            return True
        return False

    def release_location(self, location):
        
        if location in self.location_data_lock:
            try:
                self.location_data_lock[location].release()
            except RuntimeError:
                
                pass

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, thread_id, device):
        
        Thread.__init__(self, name="Device Thread %d" % thread_id)
        self.thread_id = thread_id
        self.device = device

    def stop_device(self):
        
        for _ in xrange(len(self.device.worker_pool)):
            self.device.work_queue.put(None)
        self.device.work_queue.join()
        for thread in self.device.worker_pool:
            thread.join()

    def run(self):
        
        for thread in self.device.worker_pool:
            thread.start()

        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                self.stop_device()
                break

            with self.device.script_condition:
                while not self.device.timepoint_ended:
                    self.device.script_condition.wait()

                for script in self.device.scripts:
                    self.device.work_queue.put(script)

                self.device.work_queue.join()

                self.device.timepoint_ended = False
                self.device.script_condition.notify_all()

            self.device.sync_barrier.wait()


class DeviceWorker(DeviceThread):
    

    def __init__(self, thread_id, device):
        
        super(DeviceWorker, self).__init__(thread_id, device)

    def run(self):
        while True:
            item = self.device.work_queue.get()
            if item is None:
                self.device.work_queue.task_done()
                break
            (script, location) = item

            acquired_devices = []
            script_data = []

            with self.device.acquire_stage_lock:
                if self.device.acquire_location(location):
                    acquired_devices.append(self.device)
                for device in self.device.neighbours:
                    if device.device_id != self.device.device_id:
                        if device.acquire_location(location):
                            acquired_devices.append(device)

            for device in acquired_devices:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            if len(script_data) > 0:
                result = script.run(script_data)

                for device in acquired_devices:
                    device.set_data(location, result)

            for device in acquired_devices:
                device.release_location(location)

            self.device.work_queue.task_done()
