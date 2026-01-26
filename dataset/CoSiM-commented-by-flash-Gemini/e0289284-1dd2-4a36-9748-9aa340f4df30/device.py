


from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

class ThreadPool(object):
    
    def __init__(self, thread_number, device):

        self.thread_number = thread_number
        self.queue = Queue(self.thread_number)
        self.threads = []
        self.device = device

        
        for _ in xrange(thread_number):
            self.threads.append(Thread(target=self.execute))

        
        for thread in self.threads:
            thread.start()

    def execute(self):
        
        
        neighbours, script, location = self.queue.get()

        
        while neighbours is not None \
              and script is not None \
              and location is not None:

            
            self.run(neighbours, script, location)

            
            self.queue.task_done()

            
            neighbours, script, location = self.queue.get()

        
        self.queue.task_done()

    def run(self, neighbours, script, location):
        
        script_data = []
        self.device.location_lock[location].acquire()
        
        for device in neighbours:
            if device.device_id != self.device.device_id:
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
        self.device.location_lock[location].release()

    def submit(self, neighbours, script, location):
        
        self.queue.put((neighbours, script, location))

    def wait(self):
        
        self.queue.join()

    def end(self):
        
        
        self.wait()

        
        for _ in xrange(self.thread_number):
            self.submit(None, None, None)

        
        for thread in self.threads:
            thread.join()


class Barrier(object):
    
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
        
        self.location_lock = [None] * 100
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.all_devices = None
        self.recived_flag = False

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):

        
        self.all_devices = devices

        
        
        
        if self.barrier is None:
            self.barrier = Barrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        
        
        if script is not None:
            
            if self.location_lock[location] is None:
                self.location_lock[location] = Lock()
                
                self.recived_flag = True

                
                for device_number in xrange(len(self.all_devices)):
                    self.all_devices[device_number].location_lock[location] \


                        = self.location_lock[location]

            
            self.scripts.append((script, location))
            self.script_received.set()
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
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(8, self.device)

    def run(self):

        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            while True:

                
                self.device.timepoint_done.wait()

                
                if self.device.recived_flag:

                    
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
                    self.device.recived_flag = False

                else:
                    
                    self.device.timepoint_done.clear()
                    
                    self.device.recived_flag = True
                    break

            
            self.thread_pool.wait()

            
            self.device.barrier.wait()

        
        self.thread_pool.end()
