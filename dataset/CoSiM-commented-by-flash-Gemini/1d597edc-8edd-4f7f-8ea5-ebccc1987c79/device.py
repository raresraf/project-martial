

from threading import Thread, Lock
from Queue import Queue
from barrier import Barrier
from thread_pool import ThreadPool

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        
        self.scripts = []

        
        self.scripts_queue = Queue()

        
        self.lock = {}
        
        self.global_lock = None

        
        self.barrier = None

        
        self.received_commun_data = Lock()
        self.received_commun_data.acquire()

        self.iteration_is_running = False

        self.thread_pool = ThreadPool(8, device_id)
        self.thread_pool.start()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            self.lock = {}
            self.global_lock = Lock()



            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier
                    device.lock = self.lock
                    device.global_lock = self.global_lock
                    device.received_commun_data.release()

            self.received_commun_data.release()

    def assign_script(self, script, location):
        
        if script is not None:
            with self.global_lock:
                if location not in self.lock:
                    self.lock[location] = Lock()
        self.scripts_queue.put((script, location))

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

    def acquire_location(self, location):
        
        self.lock[location].acquire()

    def release_location(self, location):
        
        self.lock[location].release()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        self.device.received_commun_data.acquire()
        supervisor = self.device.supervisor
        barrier = self.device.barrier
        thread_pool = self.device.thread_pool

        while True:
            neighbours = supervisor.get_neighbours()

            
            if neighbours is None:
                thread_pool.shutdown()
                barrier.wait()
                break

            
            if neighbours.count(self.device) == 0:
                neighbours.append(self.device)

            
            if not self.device.iteration_is_running:
                self.device.iteration_is_running = True

                for (script, location) in self.device.scripts:
                    task = (script, location, neighbours)
                    thread_pool.add_task(task)

            while True:
                
                (script, location) = self.device.scripts_queue.get()

                
                if script is None and location is None:
                    self.device.iteration_is_running = False
                    break
                else:
                    
                    task = (script, location, neighbours)
                    thread_pool.add_task(task)

                    
                    self.device.scripts.append((script, location))

            
            
            thread_pool.wait_finish()
            barrier.wait()


from threading import Thread
from Queue import Queue

class Worker(Thread):
    
    def __init__(self, worker_id, thread_pool):
        

        Thread.__init__(self, name="Worker Thread %d" % worker_id)

        self.worker_id = worker_id
        self.thread_pool = thread_pool

    def run(self):
        
        while True:
            task = self.thread_pool.get_task()

            
            
            (script, location, devices) = task

            
            
            if script is None and location is None and devices is None:
                
                self.thread_pool.task_done()

                
                break

            
            script_data = []

            
            chosen_device = devices[0]
            chosen_device.acquire_location(location)

            
            working_devices = []
            for device in devices:
                data = device.get_data(location)

                if data is not None:
                    script_data.append(data)
                    working_devices.append(device)

            if script_data != []:
                
                result = script.run(script_data)

                
                
                for device in working_devices:
                    device.set_data(location, result)

            
            chosen_device.release_location(location)

            
            self.thread_pool.task_done()

class ThreadPool(object):
    
    def __init__(self, num_threads, pool_id):
        
        self.num_threads = num_threads
        self.queue_taks = Queue()
        self.pool_id = pool_id

        self.worker = []
        for i in xrange(self.num_threads):
            self.worker.append(Worker(i, self))

    def start(self):
        
        for i in xrange(self.num_threads):
            self.worker[i].start()

    def add_task(self, task):
        
        self.queue_taks.put(task)

    def get_task(self):
        
        return self.queue_taks.get()

    def task_done(self):
        
        self.queue_taks.task_done()

    def wait_finish(self):
        
        self.queue_taks.join()

    def shutdown(self):
        

        self. wait_finish()

        
        for i in xrange(self.num_threads):
            self.add_task((None, None, None))

        
        for i in xrange(self.num_threads):
            self.worker[i].join()
