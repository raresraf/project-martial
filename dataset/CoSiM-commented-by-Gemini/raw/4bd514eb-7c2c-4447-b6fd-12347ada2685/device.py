


from threading import Event, Thread, Lock, Condition
from Queue import Queue

class ReusableBarrier(object):
    
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
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.queue = Queue()
        self.setup = Event()
        self.threads = []
        self.locations_lock = []
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            for _ in range(25):
                lock = Lock()
                self.locations_lock.append(lock)

            for device in devices:
                device.barrier = barrier
                device.locations_lock = self.locations_lock
                device.setup.set()

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
        
        self.device.setup.wait()

        for _ in range(8):
            thread = MyWorker(self.device)
            thread.start()
            self.device.threads.append(thread)

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                for thread in self.device.threads:
                    for _ in range(8):
                        self.device.queue.put(None)
                    thread.join()
                break
            self.device.timepoint_done.wait()
            self.device.barrier.wait()
            for (script, location) in self.device.scripts:
                self.device.queue.put((neighbours, location, script))

            self.device.timepoint_done.clear()
            
            self.device.barrier.wait()

class MyWorker(Thread):
    
    def __init__(self, device):
        
        Thread.__init__(self)
        self.device = device

    def run(self):
        while True:
            
            elem = self.device.queue.get()
            
            if elem is None:
                break
            
            self.device.locations_lock[elem[1]].acquire()
            script_data = []
            data = None


            for device in elem[0]:
                data = device.get_data(elem[1])
            if data is not None:
                script_data.append(data)
            
            data = self.device.get_data(elem[1])
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = elem[2].run(script_data)

                
                for device in elem[0]:
                    device.set_data(elem[1], result)
                
                self.device.set_data(elem[1], result)
            self.device.locations_lock[elem[1]].release()

            self.device.queue.task_done()
