




from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from thread_pool import ThreadPool

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        
        self.barrier = None

        
        
        self.location_locks = {}

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.barrier is None:
            
            self.barrier = ReusableBarrierCond(len(devices))

            
            for device in devices:
                device.barrier = self.barrier

                
                
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
                
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
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
    
    
    NO_CORES = 8

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.thread_pool = ThreadPool(self.device, DeviceThread.NO_CORES)

    def run(self):
        
        while True:
            

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                
                for _ in xrange(DeviceThread.NO_CORES):
                    self.thread_pool.submit_task(None, None, None)
                
                self.thread_pool.end_workers()
                break

            
            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                self.thread_pool.submit_task(script, location, neighbours)

            
            self.device.timepoint_done.clear()

            
            
            self.device.barrier.wait()




from threading import Thread
from Queue import Queue

class Worker(Thread):
    

    def __init__(self, device, task_queue):
        
        Thread.__init__(self)
        self.device = device
        self.task_queue = task_queue

    def run(self):
        while True:
            
            script, location, neighbours = self.task_queue.get()

            
            if (script is None and location is None and neighbours is None):
                self.task_queue.task_done()
                break

            
            
            with self.device.location_locks[location]:
                
                self.run_task(script, location, neighbours)

            
            self.task_queue.task_done()

    def run_task(self, script, location, neighbours):
        
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


class ThreadPool(object):
    

    def __init__(self, device, no_workers):
        
        self.device = device
        self.no_workers = no_workers
        
        self.task_queue = Queue(no_workers)
        self.workers = []
        self.initialize_workers()

    def initialize_workers(self):
        
        for _ in xrange(self.no_workers):
            self.workers.append(Worker(self.device, self.task_queue))

        for worker in self.workers:
            worker.start()

    def end_workers(self):
        
        self.task_queue.join()

        for worker in self.workers:
            worker.join()

    def submit_task(self, script, location, neighbours):
        
        self.task_queue.put((script, location, neighbours))
