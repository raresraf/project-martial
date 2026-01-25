


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.are_locks_ready = Event() 
        self.master_id = None
        self.is_master = True 
        self.barrier = None 
        self.stored_devices = [] 
        self.data_lock = [None] * 100 
        self.master_barrier = Event() 
        self.lock = Lock() 
        self.started_threads = [] 
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        for device in devices:
            if device is not None and device.master_id is not None:
                self.master_id = device.master_id
                self.is_master = False
                break

        if self.is_master is True:
            
            self.barrier = ReusableBarrierSem(len(devices))
            self.master_id = self.device_id
            for i in range(100):
                self.data_lock[i] = Lock()
            self.are_locks_ready.set()
            self.master_barrier.set()
            for device in devices:
                if device is not None:
                    device.barrier = self.barrier
                    self.stored_devices.append(device)
        else: 
            for device in devices:
                if device is not None:
                    if device.device_id == self.master_id:
                        device.master_barrier.wait()
                        if self.barrier is None:


                            self.barrier = device.barrier
                    self.stored_devices.append(device)

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            for device in self.stored_devices:
                if device.device_id == self.master_id:
                    device.are_locks_ready.wait()
            for device in self.stored_devices:


                if device.device_id == self.master_id:
                    self.data_lock = device.data_lock
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        
        self.thread.join()


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


            


            for (script, location) in self.device.scripts:
                executor = ExecutorThread(self.device, script, neighbours, location)
                self.device.started_threads.append(executor)
                executor.start()

            for executor in self.device.started_threads:
                executor.join()

            
            del self.device.started_threads[:]
            self.device.timepoint_done.clear()
            self.device.barrier.wait()


class ExecutorThread(Thread):
    

    def __init__(self, device, script, neighbours, location):
        Thread.__init__(self, name="Executor Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.location = location

    def run(self):
        self.device.data_lock[self.location].acquire()

        if self.neighbours is None:
            return

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

        self.device.data_lock[self.location].release()
