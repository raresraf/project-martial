

from threading import Event, Thread, Lock
from barrier import Barrier
from workerfactory import WorkerFactory

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
        self.locks = []
        self.barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        num_devices = len(devices)
        if self.barrier is None and self.device_id == 0:
            self.barrier = Barrier(num_devices)
            for dev in devices:
                if dev.barrier is None:
                    dev.barrier = self.barrier
        for loc in self.sensor_data:
            self.locks.append((loc, Lock()));

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            for iter in self.locks:
                if iter[0] == location:
                    iter[1].acquire();
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            for iter in self.locks:
                if iter[0] == location:
                    iter[1].release();

    def shutdown(self):
        


        self.thread.join()


class DeviceThread(Thread):
    
    num_cores = 8
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.worker_factory = WorkerFactory(DeviceThread.num_cores, device)

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            while True:
                
                if self.device.timepoint_done.wait():
                    


                    if self.device.script_received.isSet():
                        self.device.script_received.clear()
                        for (script, location) in self.device.scripts:
                            self.worker_factory.add_tasks((neighbours, script, location))
                    else:
                        
                        self.device.timepoint_done.clear()
                        
                        self.device.script_received.set()
                        break

            
            self.worker_factory.wait_for_finish()

            
            self.device.barrier.wait()

        
        self.worker_factory.shutdown()

from Queue import Queue
from threading import Thread

class WorkerFactory(object):
    
    def __init__(self, num_workers, parent_device):
        
        self.num_workers = num_workers
        self.task_queue = Queue(num_workers)
        self.worker_threads = []
        self.current_device = parent_device
        self.start_workers()

    def start_workers(self):
        
        for _ in range(0, self.num_workers):
            worker_thread = Worker(self.task_queue, self.current_device)
            self.worker_threads.append(worker_thread)
        for worker in self.worker_threads:
            worker.start()

    def add_tasks(self, necessary_data):
        
        self.task_queue.put(necessary_data)

    def wait_for_finish(self):
        
        self.task_queue.join()

    def shutdown(self):
        
        self.task_queue.join()
        for _ in xrange(self.num_workers):
            self.add_tasks((None, None, None))

        for worker in self.worker_threads:
            worker.join()

class Worker(Thread):
    
    def __init__(self, task_queue, parent_device):
        Thread.__init__(self)
        self.my_queue = task_queue
        self.current_device = parent_device

    def run(self):

        while True:
            neigh, script, location = self.my_queue.get()
            if neigh is None or script is None or location is None:
                self.my_queue.task_done()
                break

            script_data = []
            

            for device in neigh:
                if self.current_device.device_id != device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

            
            data = self.current_device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                for device in neigh:
                    if self.current_device.device_id != device.device_id:
                        device.set_data(location, result)
                
                self.current_device.set_data(location, result)
            self.my_queue.task_done()
