

from threading import Event, Thread, Lock
from pool import WorkPool
from reusable_barrier import ReusableBarrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()


        self.scripts = []

        self.timepoint_done = Event()
        self.other_devs = []
        self.slock = Lock()

        self.barrier = None
        self.process = Event()

        self.global_thread_pool = None
        self.glocks = {}

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        self.other_devs = devices
        if self.device_id == self.other_devs[0].device_id:
            locks = {}
            for loc in self.sensor_data:
                locks[loc] = Lock()
            dev_cnt = len(devices)
            self.glocks = locks
            self.barrier = ReusableBarrier(dev_cnt)
            self.global_thread_pool = WorkPool(16)
        else:
            for loc in self.sensor_data:
                self.other_devs[0].glocks[loc] = Lock()
            self.glocks = self.other_devs[0].glocks
            self.global_thread_pool = self.other_devs[0].global_thread_pool
            self.barrier = self.other_devs[0].barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            
            self.script_received.set()

    def get_data(self, location):
        
        
        ret = None
        with self.slock:


            if location in self.sensor_data:
                ret = self.sensor_data[location]
        return ret

    def set_data(self, location, data):
        
        with self.slock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        
        
        
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                if self.device.device_id is self.device.other_devs[0].device_id:
                    self.device.global_thread_pool.end()
                break

            self.device.script_received.wait()

            
            for (script, location) in self.device.scripts:
                self.device.global_thread_pool.work((self.device,
                									script,
                									location,
                									neighbours))

            self.device.global_thread_pool.finish_work()
            self.device.script_received.clear()
            self.device.barrier.wait()


from threading import Lock, Event, Semaphore, Thread

class WorkerThread(Thread):
    
    def __init__(self, i, parent_work_pool):
        
        Thread.__init__(self, name="WorkerThread%d" % i)
        self.pool = parent_work_pool

    def run(self):
        
        while True:
            self.pool.task_sign.acquire()
            if self.pool.stop:
                break
            current_task = (None, None, None, None)
            with self.pool.task_lock:
                task_count = len(self.pool.tasks_list)
                
                if task_count > 0:
                    current_task = self.pool.tasks_list[0]
                    self.pool.tasks_list = self.pool.tasks_list[1:]
                
                if task_count == 1:
                    self.pool.no_tasks.set()

            if current_task is not None:
                (current_device, script, location, neighbourhood) = current_task


                with current_device.glocks[location]:
                    common_data = []
                    
                    for neighbour in neighbourhood:
                        data = neighbour.get_data(location)
                        if data is not None:
                            common_data.append(data)
                    
                    data = current_device.get_data(location)
                    if data is not None:
                        common_data.append(data)

                    if common_data != []:
                         
                        result = script.run(common_data)
                        for neighbour in neighbourhood:
                            neighbour.set_data(location, result)
                        
                        current_device.set_data(location, result)

class WorkPool(object):
    
    def __init__(self, size):
        self.size = size

        self.tasks_list = [] 
        self.task_lock = Lock()
        self.task_sign = Semaphore(0)
        self.no_tasks = Event()
        self.no_tasks.set()
        self.stop = False

        self.workers = []
        for i in xrange(self.size):
            worker = WorkerThread(i, self)
            self.workers.append(worker)

        for worker in self.workers:
            worker.start()

    def work(self, task):
        
        with self.task_lock:
            self.tasks_list.append(task)
            self.task_sign.release()
            if self.no_tasks.is_set():
                self.no_tasks.clear()

    def finish_work(self):
        
        self.no_tasks.wait()

    def end(self):
        
        self.finish_work()
        self.stop = True
        for thread in self.workers:
            self.task_sign.release()
        for thread in self.workers:
            thread.join()
