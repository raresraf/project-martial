


from threading import Event, Thread, Condition, Lock
import Queue

WORKERS_PER_DEVICE = 8
NUM_LOCATION_LOCKS = 25

class Barrier():
    
    def __init__(self, num_threads):
        
        self.condition = Condition()
        self.count_threads = 0
        self.num_threads = num_threads

    def wait(self):
        
        self.condition.acquire()
        self.count_threads = self.count_threads + 1

        
        if self.count_threads == self.num_threads:
            self.condition.notify_all()
            self.count_threads = 0
        else:
            self.condition.wait()

        self.condition.release()

class Job():
    
    def __init__(self, location, neighbours, script):
        
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def get_location(self):
        
        return self.location

    def get_neighbours(self):
        
        return self.neighbours

    def get_script(self):
        
        return self.script

class DeviceSync(object):
    
    def __init__(self):
        
        self.barrier = None
        self.location_locks = []
        self.receive_scripts =  Event()
        self.setup = Event()

    def init_barrier(self, num_threads):
        
        self.barrier = Barrier(num_threads)

    def wait_threads(self):
        
        self.barrier.wait()

    def init_location_locks(self, num_locations):
        
        for location in range(0, num_locations):
            self.location_locks.append(Lock())

    def get_location_lock(self, location):
        
        return self.location_locks[location]



    def set_receive_scripts(self):
        
        self.receive_scripts.set()

    def wait_receive_scripts(self):
        
        self.receive_scripts.wait()

    def clear_scripts(self):
        
        self.receive_scripts.clear()

    def set_setup_event(self):
        
        self.setup.set()

    def wait_setup_event(self):
        
        self.setup.wait()

class Worker(Thread):
    
    def __init__(self, device, scripts):
        
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts

    def get_neighbours_data(self, job):
        
        
        all_data = []

        
        for device in job.neighbours:
            data = device.get_data(job.get_location())
            all_data.append(data)

        
        data = self.device.get_data(job.get_location())
        all_data.append(data)

        
        all_data = list(filter(None, all_data))

        
        return all_data

    def update_neighbours_data(self, job, new_data):
        
        for device in job.neighbours:
            device.set_data(job.get_location(), new_data)
        self.device.set_data(job.get_location(), new_data)

    def run(self):
        
        while True:
            
            job = self.scripts.get()

            
            if job.script is None:
                
                
                self.scripts.task_done()
                break

            
            with self.device.syncronization.get_location_lock(job.get_location()):
                data = self.get_neighbours_data(job)
                if data != []:
                    new_data = job.script.run(data)
                    self.update_neighbours_data(job, new_data)

            
            self.scripts.task_done()

class WorkerPool(object):
    


    def __init__(self, device, num_workers):
        
        self.device = device
        self.num_workers = num_workers

        self.scripts = Queue.Queue()
        self.workers_scripts = []

        self.start_workers()

    def start_workers(self):
        
        for worker_id in range(0, self.num_workers):
            self.workers_scripts.append(Worker(self.device, self.scripts))
            self.workers_scripts[worker_id].start()

    def join_workers(self):
        


        for worker_id in range(0, self.num_workers - 1):
            self.scripts.join()
            self.workers_scripts[worker_id].join()

    def stop_workers(self):
        
        
        for worker_id in range(0, WORKERS_PER_DEVICE):
            self.add_job(Job(None, None, None))

        self.join_workers()

    def delete_workers(self):
        
        for worker_id in range(0, self.num_workers - 1):
            del self.workers_scripts[-1]

    def add_job(self, job):
        
        self.scripts.put(job)

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.scripts = []
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.syncronization = DeviceSync()
        self.worker_pool = WorkerPool(self, WORKERS_PER_DEVICE)
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        
        
        if self.device_id == len(devices) - 1:
            self.syncronization.init_barrier(len(devices))
            self.syncronization.init_location_locks(NUM_LOCATION_LOCKS)



            for device in devices:
                device.syncronization.barrier = self.syncronization.barrier
                device.syncronization.location_locks = self.syncronization.location_locks
                device.syncronization.set_setup_event()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.syncronization.set_receive_scripts()

    def add_job(self, job):
        
        self.worker_pool.add_job(job)

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
        


        self.device.syncronization.wait_setup_event()

        
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            
            if neighbours is None:
                self.device.worker_pool.stop_workers()
                break

            
            self.device.syncronization.wait_threads()
            self.device.syncronization.wait_receive_scripts()

            
            for (script, location) in self.device.scripts:
                self.device.add_job(Job(location, neighbours, script))

            
            self.device.syncronization.wait_threads()
            self.device.syncronization.clear_scripts()
