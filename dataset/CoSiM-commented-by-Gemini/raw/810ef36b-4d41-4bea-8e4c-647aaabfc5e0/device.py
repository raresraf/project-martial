


import multiprocessing
from threading import Event, Thread, Lock
from threadpool import ThreadPool
from reusablebarrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.time_point_done = Event()
        self.nr_cpu = multiprocessing.cpu_count()
        self.thread = DeviceThread(self, self.nr_cpu)

        self.locations_lock_set = None
        self.barrier = None

        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        


        if self.device_id == 0:
            my_barrier = ReusableBarrier(len(devices))

            for dev in devices:
                
                dev.barrier = my_barrier

            
            locations_lock_set = {}

            for dev in devices:
                
                for location in dev.sensor_data:
                    if location not in locations_lock_set:
                        locations_lock_set[location] = Lock()

            
            for dev in devices:
                dev.locations_lock_set = locations_lock_set

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            
            self.time_point_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, nr_cpu):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.nr_cpu = nr_cpu
        self.pool = ThreadPool(nr_cpu, device)

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            
            if neighbours is None:
                
                for _ in xrange(self.nr_cpu):
                    self.pool.add_task((True, None, None))

                
                self.pool.wait_workers()
                break

            
            
            self.device.time_point_done.wait()

            
            self.device.time_point_done.clear()

            
            for my_script in self.device.scripts:
                self.pool.add_task((False, my_script, neighbours))

            
            self.pool.wait_completion()

            
            self.device.barrier.wait()


from threading import Thread
from Queue import Queue


class ThreadPool(object):
    

    def __init__(self, num_threads, device):
        
        self.queue = Queue(num_threads)
        self.device = device
        self.workers = []
        for _ in xrange(num_threads):
            adt = AuxiliaryDeviceThread(self.device, self.queue)
            self.workers.append(adt)

    def add_task(self, info):
        
        self.queue.put(info)

    def wait_completion(self):
        
        self.queue.join()

    def wait_workers(self):
        
        for adt in self.workers:
            adt.join()


class AuxiliaryDeviceThread(Thread):
    

    def __init__(self, device, queue):
        
        Thread.__init__(self)
        self.queue = queue
        self.device = device
        self.daemon = True
        self.start()

    def run(self):

        while True:
            
            can_finish, got_script, neighbours = self.queue.get()

            if can_finish:  
                
                break

            
            (script, location) = got_script

            
            
            self.device.locations_lock_set[location].acquire()

            
            script_data = []

            
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            
            data = self.device.get_data(location)


            if data is not None:
                script_data.append(data)

            if script_data:
                
                result = script.run(script_data)

                
                for device in neighbours:
                    device.set_data(location, result)

                
                self.device.set_data(location, result)

            
            self.device.locations_lock_set[location].release()

            
            self.queue.task_done()
