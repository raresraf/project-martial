


from threading import Lock, Event, Thread
from barrier import ReusableBarrierCond
from workerpool import WorkerPool


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        self.barrier = None
        self.locks = None

        self.thread = DeviceThread(self)
        self.thread.start()


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:

            
            num_threads = len(devices)
            barrier = ReusableBarrierCond(num_threads)
            location_locks = {}

            
            for device in devices:
                for location in device.sensor_data:
                    if location not in location_locks:
                        location_locks[location] = Lock()

            
            if self.barrier is None:
                self.barrier = barrier
                self.locks = location_locks

            
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
                if device.locks is None:
                    device.locks = location_locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        data = None

        if location in self.sensor_data:
            data = self.sensor_data[location]
        return data

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.pool = WorkerPool(8, self.device)

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                
                for _ in xrange(self.pool.max_threads):
                    self.pool.submit_work(None, None, None)
                self.pool.wait_completion()
                break

            
            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                self.pool.submit_work(neighbours, script, location)

            
            self.pool.wait_completion()

            
            self.device.barrier.wait()

            
            self.device.timepoint_done.clear()

        
        self.pool.end_threads()


from threading import Thread
from Queue import Queue


class WorkerPool(object):
    

    def __init__(self, no_workers, device):
        
        self.max_threads = no_workers
        self.queue = Queue(no_workers)
        self.device = device
        self.thread_list = []
        for _ in range(no_workers):
            thread = WorkerThread(self.device, self.queue)
            self.thread_list.append(thread)
            thread.start()

    def submit_work(self, neighbours, script, location):
        
        self.queue.put((neighbours, script, location))

    def wait_completion(self):
        
        self.queue.join()

    def end_threads(self):
        
        for thread in self.thread_list:
            thread.join()
        self.thread_list = []


class WorkerThread(Thread):
    

    def __init__(self, device, tasks):
        
        Thread.__init__(self)
        self.device = device
        self.tasks = tasks

    def run(self):
        
        while True:
            
            neighbours, script, location = self.tasks.get()

            
            if neighbours is None and script is None and location is None:
                self.tasks.task_done()
                return

            
            with self.device.locks[location]:
                
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

            
            self.tasks.task_done()
