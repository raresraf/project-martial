


from threading import Event, Thread, Semaphore, Lock
from Queue import Queue

class Device(object):
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.time_point_barrier = None
        self.location_semaphore_dict = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            barrier = ReusableBarrierSem(len(devices))
            self.time_point_barrier = barrier
            for dev in devices:
                if dev.time_point_barrier is None:
                    dev.time_point_barrier = barrier
            location_set = set()
            for dev in devices:
                for location in dev.sensor_data:
                    location_set.add(location)
            loc_dict = {}
            for loc in location_set:
                loc_dict[loc] = Semaphore(1)
            for dev in devices:
                dev.location_semaphore_dict = loc_dict

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class InnerThread(Thread):
    
    def __init__(self, device, barrier, queue):
        
        Thread.__init__(self)
        self.device = device
        self.barrier = barrier
        self.queue = queue
        self.neighbours = []

    def run(self):
        while True:
            
            script = self.queue.get()
            if script[0] == "exit":
                self.barrier.wait()
                break
            if script[0] == "done":
                self.barrier.wait()
                continue
            if script[0] == "neighbours":
                self.neighbours = script[1]
                self.barrier.wait()
                continue
            script_improved = script[0]
            location = script[1]
            script_data = []
            
            self.device.location_semaphore_dict[location].acquire()
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script_improved.run(script_data)
                
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)
            self.device.location_semaphore_dict[location].release()

class DeviceThread(Thread):
    
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.cores_number = 8
        self.threads_barrier = ReusableBarrierSem(self.cores_number)
        self.scripts_queue = Queue()
        self.thread_list = []

    def run(self):
        for _ in range(self.cores_number):
            inner_t = InnerThread(self.device, self.threads_barrier,\
            self.scripts_queue)
            self.thread_list.append(inner_t)
        for thread in self.thread_list:
            thread.start()
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                for thread in self.thread_list:
                    self.scripts_queue.put(("exit", None))
                break
            self.device.timepoint_done.wait()
            for _ in range(self.cores_number):
                self.scripts_queue.put(("neighbours", neighbours))
            for pair in self.device.scripts:
                self.scripts_queue.put(pair)
            for _ in range(self.cores_number):
                self.scripts_queue.put(("done", None))
            self.device.timepoint_done.clear()
            self.device.time_point_barrier.wait()


        for thread in self.thread_list:
            thread.join()

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
