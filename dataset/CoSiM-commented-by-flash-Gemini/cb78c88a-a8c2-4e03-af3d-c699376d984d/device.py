



from threading import Event, Thread, Lock, BoundedSemaphore
from barrier import ReusableBarrier


class Device(object):
    

    timepoint_barrier = None
    barrier_lock = Lock()

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.max_threads_semaphore = BoundedSemaphore(8)

    def __str__(self):
        
        return "Device %d" % self.device_id

    @staticmethod
    def setup_devices(devices):
        
        
        

        
        if Device.timepoint_barrier is None:
            Device.barrier_lock.acquire()


            if Device.timepoint_barrier is None:
                Device.timepoint_barrier = ReusableBarrier(len(devices))
            Device.barrier_lock.release()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
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

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            threads = []
            
            for (script, location) in self.device.scripts:
                self.device.max_threads_semaphore.acquire()

                
                worker_thread = ScriptWorkerThread(self.device, neighbours, location, script)
                threads.append(worker_thread)
                worker_thread.start()

            
            for thread in threads:
                thread.join()

            
            self.device.timepoint_done.clear()

            
            Device.timepoint_barrier.wait()


class ScriptWorkerThread(Thread):

    

    locations_lock = {}

    def __init__(self, device, neighbours, location, script):


        super(ScriptWorkerThread, self).__init__()
        self.device = device
        self.neighbours = neighbours
        self.location = location
        self.script = script

        
        if location not in ScriptWorkerThread.locations_lock:
            ScriptWorkerThread.locations_lock[location] = Lock()

    def run(self):

        
        ScriptWorkerThread.locations_lock[self.location].acquire()

        script_data = []
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        ScriptWorkerThread.locations_lock[self.location].release()

        
        self.device.max_threads_semaphore.release()
