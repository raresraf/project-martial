


from threading import Event, Thread, Lock, Semaphore
from multiprocessing import cpu_count
from Queue import Queue


class ReusableBarrierSem(object):
    

    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads

        
        self.counter_lock = Lock()

        
        self.threads_sem1 = Semaphore(0)

        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase1()
        self.phase2()

    def phase1(self):
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        
        
        
        self.scripts = []

        
        
        self.timepoint_done = Event()

        
        self.neighbours = []

        
        
        
        self.num_of_threads = cpu_count()
        if self.num_of_threads < 8:
            self.num_of_threads = 8

        
        
        
        
        
        
        self.tasks = Queue()

        
        
        
        self.semaphore = Semaphore(0)

        
        
        
        self.num_locations = self.supervisor.supervisor.testcase.num_locations
        self.lock_locations = []

        
        
        
        
        
        
        
        self.lock_queue = Lock()

        
        self.barrier = ReusableBarrierSem(self.num_of_threads)

        
        
        self.global_barrier = ReusableBarrierSem(0)

        
        self.pool = Pool(self)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        
        
        
        
        if self.device_id == 0:
            self.global_barrier = ReusableBarrierSem(len(devices))

            for _ in range(self.num_locations):
                self.lock_locations.append(Semaphore(1))

            for device in devices:
                device.global_barrier = self.global_barrier
                device.lock_locations = self.lock_locations

    def assign_script(self, script, location):
        
        if script is not None:
            
            
            
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] \
            if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        

        
        self.pool.shutdown()


class Pool(object):
    

    def __init__(self, device):
        

        
        self.device = device

        
        self.thread_list = []

        
        for i in range(self.device.num_of_threads):
            self.thread_list.append(DeviceThread(self.device, i))

        
        for thread in self.thread_list:
            thread.start()

    def add_task(self, task):
        
        self.device.tasks.put(task)
        self.device.semaphore.release()

    def shutdown(self):
        
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        
        self.thread_id = thread_id

        
        self.script = None
        self.location = None

        
        self.script_data = []

        
        self.data = None

        
        self.result = None

    def run(self):
        

        
        while True:

            
            
            if self.thread_id == 0:
                
                self.device.timepoint_done.clear()

                
                self.device.neighbours = \
                                self.device.supervisor.get_neighbours()

            
            
            self.device.barrier.wait()

            
            
            if self.device.neighbours is None:
                break

            
            
            
            
            if self.thread_id == 0:
                
                self.device.timepoint_done.wait()

                
                for (script, location) in self.device.scripts:
                    self.device.pool.add_task((script, location))

                
                
                
                
                
                
                
                
                

                for _ in range(self.device.num_of_threads):
                    self.device.semaphore.release()


            
            while True:

                
                self.device.semaphore.acquire()

                
                
                

                with self.device.lock_queue:
                    if not self.device.tasks.empty():
                        
                        (self.script, self.location) = self.device.tasks.get()
                    
                    else:
                        break

                
                
                
                self.device.lock_locations[self.location].acquire()

                self.script_data = []

                


                for device in self.device.neighbours:
                    self.data = device.get_data(self.location)
                    if self.data is not None:
                        self.script_data.append(self.data)

                
                self.data = self.device.get_data(self.location)
                if self.data is not None:
                    self.script_data.append(self.data)

                
                if self.script_data != []:
                    self.result = self.script.run(self.script_data)

                    
                    for device in self.device.neighbours:


                        device.set_data(self.location, self.result)

                    
                    self.device.set_data(self.location, self.result)

                
                self.device.lock_locations[self.location].release()

            
            
            if self.thread_id == 0:
                self.device.global_barrier.wait()

            
            
            
            self.device.barrier.wait()
