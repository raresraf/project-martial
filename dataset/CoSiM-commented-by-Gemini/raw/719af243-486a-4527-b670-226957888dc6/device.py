


from threading import Event, Thread, RLock
from Queue import Queue
from barrier import ReusableBarrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []


        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.locations_locks = {}
        self.reusable_barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id != 0:
            return
        locations_locks = {}
        reusable_barrier = ReusableBarrier(len(devices))
        for device in devices:
            device.locations_locks = locations_locks
            device.reusable_barrier = reusable_barrier

    def assign_script(self, script, location):
        

        
        if location not in self.locations_locks:
            self.locations_locks[location] = RLock()

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

        self.script_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class WorkerThread(Thread):
    

    def __init__(self, master, index):
        
        Thread.__init__(self, name="Worker Thread %d" % index)
        self.master = master
        self.index = index

    def get_script_data(self, location, neighbours):
        

        
        data = [d for d in [n.get_data(location) for n in neighbours] if d is not None]

        
        my_data = self.master.device.get_data(location)
        if my_data is not None:
            data.append(my_data)

        return data

    def broadcast_result(self, location, result, neighbours):
        
        self.master.device.set_data(location, result)
        for device in neighbours:
            device.set_data(location, result)

    def run(self):
        while True:
            
            task = self.master.queue.get()
            if task is None:
                break

            
            script = task[0]
            location = task[1]
            neighbours = task[2]

            
            with self.master.device.locations_locks[location]:
                script_data = self.get_script_data(location, neighbours)
                if script_data != []:
                    result = script.run(script_data)
                    self.broadcast_result(location, result, neighbours)

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue()

    def run(self):
        
        threads_number = 8
        workers = []
        for i in range(0, threads_number):
            workers.append(WorkerThread(self, i))

        
        for worker in workers:
            worker.start()

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                
                for i in range(0, threads_number):
                    self.queue.put(None)
                break

            
            while not self.device.timepoint_done.isSet():
                self.device.script_received.wait()
            self.device.script_received.clear()

            
            for (script, location) in self.device.scripts:
                self.queue.put((script, location, neighbours))

            
            self.device.reusable_barrier.wait()
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

        
        for worker in workers:
            worker.join()
            