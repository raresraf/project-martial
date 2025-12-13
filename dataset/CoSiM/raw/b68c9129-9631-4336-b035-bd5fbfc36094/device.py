


from threading import Event, Thread, Lock
from barrier import Barrier
from thread_pool import ThreadPool


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data


        self.supervisor = supervisor

        self.barrier = None
        self.scripts = []
        self.timepoint_done = Event()

        
        self.locks = {}

        for location in sensor_data:
            self.locks[location] = Lock()

        
        self.scripts_available = False

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            barrier = Barrier(len(devices))
            self.barrier = barrier
            self.send_barrier(devices, barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        
        self.barrier = barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
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
    
    NR_THREADS = 8

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        self.thread_pool = ThreadPool(self.NR_THREADS)

    def run(self):
        self.thread_pool.set_device(self.device)

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            while True:
                self.device.timepoint_done.wait()
                if self.device.scripts_available:
                    self.device.scripts_available = False

                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task((neighbours, location, script))
                else:
                    self.device.timepoint_done.clear()
                    self.device.scripts_available = True
                    break

            
            self.thread_pool.wait()

            
            self.device.barrier.wait()

        
        self.thread_pool.finish()


from threading import Thread
from Queue import Queue


class ThreadPool(object):
    
    def __init__(self, nr_threads):
        


        self.device = None

        self.queue = Queue(nr_threads)
        self.thread_list = []

        self.create_threads(nr_threads)
        self.start_threads()

    def create_threads(self, nr_threads):
        
        for _ in xrange(nr_threads):
            thread = Thread(target=self.execute_task)
            self.thread_list.append(thread)

    def start_threads(self):
        
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].start()

    def set_device(self, device):
        
        self.device = device

    def submit_task(self, task):
        
        self.queue.put(task)

    def execute_task(self):
        

        while True:
            task = self.queue.get()

            neighbours = task[0]
            script = task[2]

            if script is None and neighbours is None:
                self.queue.task_done()
                break

            self.run_script(task)
            self.queue.task_done()

    def run_script(self, task):
        

        neighbours, location, script = task
        script_data = []

        
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            
            result = script.run(script_data)

            
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

            
            self.device.set_data(location, result)

    def wait(self):
        
        self.queue.join()

    def finish(self):
        
        self.wait()

        
        for _ in xrange(len(self.thread_list)):
            self.submit_task((None, None, None))

        
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].join()
