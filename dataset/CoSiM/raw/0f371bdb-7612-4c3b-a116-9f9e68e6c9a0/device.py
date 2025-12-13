


from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierCond


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.data_lock = None

        self.scripts = []
        self.devices = []
        self.neighbours = []

        
        self.first_script = -1
        self.last_script = -1
        self.script_available = Condition()

        self.timepoint_done = Event()
        
        self.leader_id = self.device_id
        self.timepoint_sync_barrier = None
        self.timepoint_barrier_set = Event()

        
        self.barrier = ReusableBarrierCond(8)

        self.threads = []
        for i in xrange(8):
            self.threads.append(DeviceThread(self, i))

        for i in xrange(8):
            self.threads[i].start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def set_neighbours(self):
        
        self.neighbours = self.supervisor.get_neighbours()

    def setup_devices(self, devices):
        
        self.devices = devices
        for dev in self.devices:
            dev_id = dev.device_id
            if dev_id < self.leader_id:
                self.leader_id = dev_id

        if self.leader_id == self.device_id:
            self.data_lock = {}
            self.timepoint_sync_barrier = ReusableBarrierCond(len(devices))
            for dev in devices:
                dev.timepoint_sync_barrier = self.timepoint_sync_barrier
                dev.data_lock = self.data_lock
                dev.timepoint_barrier_set.set()
        else:
            self.timepoint_barrier_set.wait()


    def assign_script(self, script, location):
        
        if script is not None:
            self.script_available.acquire()

            if location not in self.data_lock:
                self.data_lock[location] = Lock()

            self.last_script = self.last_script + 1
            self.scripts.append((script, location))
            self.script_available.notify()

            self.script_available.release()
        else:
            self.script_available.acquire()

            self.timepoint_done.set()
            self.script_available.notify_all()

            self.script_available.release()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def end_timepoint(self):
        
        self.timepoint_sync_barrier.wait()
        self.first_script = -1
        self.timepoint_done.clear()

    def shutdown(self):
        
        for i in xrange(8):
            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, my_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


        self.my_id = my_id

    def run(self):
        while True:
            if self.my_id == 0:
                self.device.set_neighbours()

            self.device.barrier.wait()
            neighbours = self.device.neighbours

            if neighbours is None:
                break
            while True:
                script = None
                location = None

                self.device.script_available.acquire()
                if self.device.first_script == self.device.last_script:
                    while not self.device.timepoint_done.is_set():
                        self.device.script_available.wait()
                        if self.device.first_script < self.device.last_script:
                            self.device.first_script = self.device.first_script + 1
                            (script, location) = self.device.scripts[self.device.first_script]
                else:
                    self.device.first_script = self.device.first_script + 1
                    (script, location) = self.device.scripts[self.device.first_script]
                self.device.script_available.release()

                if location is None:
                    break

                with self.device.data_lock[location]:
                    
                    script_data = []
                    
                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        
                        result = script.run(script_data)

                        
                        for device in neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)

            self.device.barrier.wait()
            if self.my_id == 0:
                self.device.end_timepoint()
            self.device.barrier.wait()


