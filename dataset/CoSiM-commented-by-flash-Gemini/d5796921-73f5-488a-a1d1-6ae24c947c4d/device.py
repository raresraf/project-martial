


from threading import Event, Thread, Lock, Condition
import Queue

class ReusableBarrier():
    
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        
        self.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
        
    def acquire(self):
        
        self.cond.acquire()
              
class Worker(Thread):
    
    def __init__(self, scripts_buffer, device):
        
        Thread.__init__(self)
        self.device = device
        self.script_buffer = scripts_buffer
        
    def get_script_data(self, job):
        
        script_data = []
        for device in job.neighbours:
            data = device.get_data(job.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(job.location)
    
        if data is not None:
            script_data.append(data)
        return script_data
        
    def update_data_on_neighbours(self, job, result):
        
        for device in job.neighbours:
            device.set_data(job.location, result)
        self.device.set_data(job.location, result)
    
    def run(self):


        while True:
        
            job = self.script_buffer.get()
            
            if job.script is None:
                self.script_buffer.task_done()
                break
            
            
            with self.device.sync.get_location_lock(job.location):
                script_data = self.get_script_data(job)
                
                if script_data != []:
                    
                    result = job.script.run(script_data)    
                    self.update_data_on_neighbours(job, result)
            
            
            self.script_buffer.task_done()      

class WorkerPool(object):
    
    


    def __init__(self, workers, device):
        
        self.workers = workers
        self.workers_scripts = []
        self.scripts_buffer = Queue.Queue()
        self.device = device
        self.start_workers()
        
    def start_workers(self):
        
        for i in range(0, self.workers):
            self.workers_scripts.append(Worker(self.scripts_buffer, 
                                               self.device))
            self.workers_scripts[i].start()
            
    def add_job(self, job):
        
        self.scripts_buffer.put(job)

    def delete_workers(self):
        
        for _ in (0, self.workers-1):
            del self.workers_scripts[-1]
            
    def join_workers(self):
        


        for i in (0, self.workers-1):
            self.scripts_buffer.join()
            self.workers_scripts[i].join()
            
    def make_workers_stop(self):
        
        for _ in range(0, 8):
            self.add_job(Job(None, None, None))
        self.join_workers()
        
class Job():
    
    def __init__(self, neighbours, script, location):
        
        self.neighbours = neighbours
        self.script = script
        self.location = location

    def get_neighbours(self):
        
        return self.neighbours
    
    def get_script(self):
        
        return self.script
     
class DeviceSync(object):
    
    def __init__(self):
        
        self.setup = Event()
        self.scripts_received = Event()
        self.location_locks = []
        self.barrier = None
        
    def init_location_locks(self, locations):
        
        for _ in range(0, locations):
            self.location_locks.append(Lock())
            
    def init_barrier(self, threads):
        
        self.barrier = ReusableBarrier(threads)
        
    def set_setup_event(self):
        
        self.setup.set()
        
    def wait_setup_event(self):
        
        self.setup.wait()
        
    def set_scripts_received(self):
        
        self.scripts_received.set()
        
    def wait_scripts_received(self):
        
        self.scripts_received.wait()
        


    def clear_scripts_received(self):
        
        self.scripts_received.clear()
        
    def wait_threads(self):
        
        self.barrier.wait()
        
    def get_location_lock(self, location):
        
        return self.location_locks[location]
        
class Device(object):
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        
        self.sync = DeviceSync()
        self.worker_pool = WorkerPool(8, self)
        
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == len(devices)-1:
            self.sync.init_location_locks(25)
            self.sync.init_barrier(len(devices))


            for device in devices:
                device.sync.barrier = self.sync.barrier
                device.sync.location_locks = self.sync.location_locks
                device.sync.set_setup_event()
            
    def add_job(self, job):
        
        self.worker_pool.add_job(job)
        
    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.sync.set_scripts_received()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

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
        


        self.device.sync.wait_setup_event()
        while True: 
            
            neighbours = self.device.supervisor.get_neighbours()
            
            if neighbours is None:
                self.device.worker_pool.make_workers_stop()
                break
            
            
            self.device.sync.wait_threads()
            self.device.sync.wait_scripts_received()
            
            
            for (script, location) in self.device.scripts:
                self.device.add_job(Job(neighbours, script, location))
            
            
            self.device.sync.wait_threads()
            self.device.sync.clear_scripts_received()