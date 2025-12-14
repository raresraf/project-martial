

from threading import Event, Thread, Lock, Condition


class ReusableBarrierCond:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    
    barrier = None

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()


        self.scripts = []
        self.timepoint_done = Event()
        self.set_data_lock = Lock()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        for device in devices:
            if device.device_id == 0:
                self.barrier = ReusableBarrierCond(len(devices))
            else:
                self.barrier = devices[0].barrier

        pass

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        with self.set_data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class ComputationThread(Thread):
    

    def __init__(self, device_thread, neighbours, script_data):
        Thread.__init__(self, name="Worker %s" % device_thread.name)
        self.device_thread = device_thread
        self.neighbours = neighbours
        self.script = script_data[0]
        self.location = script_data[1]

    def run(self):
        script_data = []
        
        for device in self.neighbours:


            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        data = self.device_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            


            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)

            
            self.device_thread.device.set_data(self.location, result)


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

            
            local_threads = []
            for script_data in self.device.scripts:
                worker = ComputationThread(self, neighbours, script_data)
                worker.start()
                local_threads.append(worker)

            for worker in local_threads:
                worker.join()

            
            self.device.timepoint_done.clear()

            
            self.device.barrier.wait()
