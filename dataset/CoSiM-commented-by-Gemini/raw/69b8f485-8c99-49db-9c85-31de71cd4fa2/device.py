


from Queue import Queue
from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        
        
        
        self.barrier = None
        self.queue = Queue()
        self.workers = [WorkerThread(self) for _ in range(8)]

        
        self.thread = DeviceThread(self)
        self.thread.start()

        
        for thread in self.workers:
            thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:

            
            barrier = ReusableBarrier(len(devices))

            
            locks = {}

            
            
            
            
            for device in devices:
                for location in device.sensor_data:
                    if not location in locks:
                        locks[location] = Lock()

            
            
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class WorkerThread(Thread):
    

    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

    def run(self):
        
        while True:
            item = self.device.queue.get()
            if item is None:
                break

            (script, location) = item

            with self.device.locks[location]:
                script_data = []

                
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    


                    for device in self.device.neighbours:
                        device.set_data(location, result)

                    
                    self.device.set_data(location, result)

            self.device.queue.task_done()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device

    def run(self):
        while True:

            
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                break

            self.device.timepoint_done.wait()


            self.device.timepoint_done.clear()

            
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location))

            
            self.device.queue.join()

            
            self.device.barrier.wait()

        
        for _ in range(8):
            self.device.queue.put(None)

        
        for thread in self.device.workers:
            thread.join()
