


from threading import Thread, Semaphore, Condition
from pool_of_threads import ThreadPool

class Sem(object):
    

    def __init__(self, devices):
        
        self.location_semaphore = {}
        for device in devices:
            for location in device.sensor_data:
                if location not in self.location_semaphore:
                    self.location_semaphore[location] = Semaphore(value=1)

    def acquire(self, location):
        
        self.location_semaphore[location].acquire()

    def release(self, location):
        
        self.location_semaphore[location].release()


class ReusableBarrierCond(object):
    

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
        self.scripts = []

        self.barrier = None
        self.location_semaphore = None
        self.timepoint_done = False

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            self.location_semaphore = Sem(devices)

            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier
                    device.location_semaphore = self.location_semaphore

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done = True

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
        self.thread_pool = ThreadPool(8, device)

    def run(self):

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            while True:

                
                if self.device.timepoint_done:


                    self.device.timepoint_done = False
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
                    break

            
            self.thread_pool.wait_threads()

            
            self.device.barrier.wait()

        
        self.thread_pool.end_threads()


from Queue import Queue
from threading import Thread

class ThreadPool(object):
    
    def __init__(self, threads_count, device):
        
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = device
        self.create_and_start_worker_threads(threads_count)

    def create_and_start_worker_threads(self, threads_count):
        


        for _ in range(threads_count):
            thread = Thread(target=self.do_job)
            self.threads.append(thread)

        for thread in self.threads:
            thread.start()

    def do_job(self):
        
        while True:

            neighbours, script, location = self.queue.get()

            if neighbours is None and script is None and location is None:
                self.queue.task_done()
                return

            
            script_data = []
            self.device.location_semaphore.acquire(location)
            
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
            self.device.location_semaphore.release(location)

            self.queue.task_done()

    def submit(self, neighbours, script, location):
        
        self.queue.put((neighbours, script, location))

    def wait_threads(self):
        
        self.queue.join()

    def end_threads(self):
        
        for _ in range(len(self.threads)):
            self.submit(None, None, None)

        self.wait_threads()

        for thread in self.threads:
            thread.join()
