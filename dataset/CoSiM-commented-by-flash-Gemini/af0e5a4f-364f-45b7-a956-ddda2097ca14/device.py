


from threading import Thread, Condition, Semaphore
from barrier import Barrier
from threadpool import ThreadPool

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.data_semaphores = {loc : Semaphore(1) for loc in sensor_data}
        self.scripts = []

        self.new_script = False
        self.timepoint_end = False
        self.cond = Condition()

        self.barrier = None
        self.supervisor = supervisor
        self.thread = DeviceThread(self)

        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            
            self.barrier = Barrier(len(devices))
            for neigh in devices:
                if neigh.device_id != self.device_id:
                    neigh.set_barrier(self.barrier)

    def set_barrier(self, barrier):
        
        self.barrier = barrier

    def assign_script(self, script, location):
        
        with self.cond:
            if script is not None:
                self.scripts.append((script, location))
                self.new_script = True
            else:
                self.timepoint_end = True
            self.cond.notifyAll()

    def timepoint_ended(self):
        
        with self.cond:
            while not self.new_script and \
                  not self.timepoint_end:
                self.cond.wait()

            if self.new_script:
                self.new_script = False
                return False
            else:
                self.timepoint_end = False
                self.new_script = len(self.scripts) > 0
                return True

    def get_data(self, location):
        


        if location in self.sensor_data:
            self.data_semaphores[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_semaphores[location].release()

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    @staticmethod
    def run_script(own_device, neighbours, script, location):
        
        script_data = []

        
        for device in neighbours:
            if device is own_device:
                continue
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        
        data = own_device.get_data(location)


        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data)

            
            for device in neighbours:
                if device is not own_device:
                    device.set_data(location, result)

            
            own_device.set_data(location, result)

    def run(self):
        
        
        
        pool_size = 8
        pool = ThreadPool(pool_size)

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            
            offset = 0
            while not self.device.timepoint_ended():
                scripts = self.device.scripts[offset:]
                for (script, location) in scripts:
                    pool.add_task(DeviceThread.run_script, self.device,
                                  neighbours, script, location)

                
                offset = len(scripts)

            
            pool.wait()

            
            self.device.barrier.wait()

        
        pool.terminate()


from Queue import Queue
from threading import Thread

class Worker(Thread):
    

    def __init__(self, tasks):
        
        Thread.__init__(self)
        self.tasks = tasks

    def run(self):
        
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except ValueError:
                return
            finally:
                self.tasks.task_done()

class ThreadPool(object):
    

    def __init__(self, num_threads):
        
        self.tasks = Queue(num_threads)
        self.workers = [Worker(self.tasks) for _ in range(num_threads)]

        for worker in self.workers:
            worker.start()

    def add_task(self, func, *args, **kargs):
        
        self.tasks.put((func, args, kargs))

    def wait(self):
        
        self.tasks.join()

    def terminate(self):
        
        self.wait()

        def raising_dummy():
            
            raise ValueError

        for _ in range(len(self.workers)):
            self.add_task(raising_dummy)
        for worker in self.workers:
            worker.join()
