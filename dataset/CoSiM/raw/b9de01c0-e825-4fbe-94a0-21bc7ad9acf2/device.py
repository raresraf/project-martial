


from threading import Event, Thread, Lock, Semaphore
from Queue import Queue


class ThreadPool(object):
    
    def __init__(self, num_threads, device):
        self.__device = device
        self.__queue = Queue(num_threads)
        self.__threads = [Thread(target=self.work) for _ in range(num_threads)]

        for thread in self.__threads:
            thread.start()

    def work(self):
        
        while True:
            script, location, neighbours = self.__queue.get()

            if not script and not neighbours:
                self.__queue.task_done()
                break

            script_data = []

            
            for device in neighbours:
                if self.__device.device_id != device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

            
            data = self.__device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                for device in neighbours:
                    if self.__device.device_id != device.device_id:
                        device.set_data(location, result)

                
                self.__device.set_data(location, result)
            self.__queue.task_done()

    def add_tasks(self, scripts, neighbours):
        
        for script, location in scripts:
            self.__queue.put((script, location, neighbours))

    def wait_threads(self):
        
        self.__queue.join()

    def stop_threads(self):
        
        self.__queue.join()

        for thread in self.__threads:
            self.__queue.put((None, None, None))



        for thread in self.__threads:
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


class Device(object):
    
    num_threads = 8

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.script_received.clear()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.locks = dict()
        for loc in sensor_data:
            self.locks[loc] = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]

        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.pool = ThreadPool(Device.num_threads, device)

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            while True:
                if self.device.script_received.is_set():
                    self.pool.add_tasks(self.device.scripts, neighbours)
                    self.device.script_received.clear()

                if self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
                    self.device.script_received.set()
                    break

            self.pool.wait_threads()
            self.device.barrier.wait()

        self.pool.stop_threads()
