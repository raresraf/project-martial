


from threading import Event, Thread, Lock
from operator import attrgetter
from barrier import ReusableBarrierCond
from pool import ThreadPool

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()
        self.scripts = []

        self.thread = DeviceThread(self)
        self.other_devices = []

        
        
        self.gdevice = None
        self.gid = None

        
        self.glocks = {}

        
        self.barrier = None

        
        self.threadpool = None

        
        self.nthreads = 8

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.other_devices = devices

        
        self.gdevice = min(devices, key=attrgetter('device_id'))
        self.gid = self.gdevice.device_id

        
        if self.device_id == self.gid:
            
            list_loc = []
            for dev in self.other_devices:
                for key, _ in dev.sensor_data.iteritems():
                    list_loc.append(key)
            list_nodup = list(set(list_loc))

            
            locks = {}
            for loc in list_nodup:
                locks[loc] = Lock()
            self.glocks = locks
            
            self.barrier = ReusableBarrierCond(len(self.other_devices))

        self.threadpool = ThreadPool(self.nthreads)

        
        self.thread.start()


    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        

        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
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
                self.device.threadpool.end()
                break

            
            self.device.script_received.wait()

            
            for (script, location) in self.device.scripts:
                self.device.threadpool.add_task((self.device, script, location, neighbours))
            self.device.threadpool.finish_tasks()

            
            self.device.script_received.clear()

            
            self.device.gdevice.barrier.wait()


from threading import Lock, Thread
from Queue import Queue

class WorkerThread(Thread):
    

    def __init__(self, threadpool):
        
        Thread.__init__(self, name="worker")
        self.threadpool = threadpool
        self.start()

    def run(self):
        while True:

            
            if self.threadpool.stop:
                break

            current_task = None

            with self.threadpool.task_lock:
                if self.threadpool.tasks.qsize() > 0:


                    current_task = self.threadpool.tasks.get_nowait()

            
            if current_task is not None:
                (device, script, location, neighbours) = current_task

                with device.gdevice.glocks[location]:
                    script_data = []
                    
                    for dev in neighbours:
                        data = dev.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        
                        result = script.run(script_data)
                        
                        for dev in neighbours:
                            dev.set_data(location, result)
                        
                        device.set_data(location, result)

                
                self.threadpool.tasks.task_done()


class ThreadPool(object):
    

    def __init__(self, size):
        
        self.size = size

        
        self.tasks = Queue()

        
        self.workers = []

        
        self.task_lock = Lock()

        
        self.stop = False

        for _ in xrange(self.size):
            self.workers.append(WorkerThread(self))

    def add_task(self, task):
        
        with self.task_lock:
            self.tasks.put(task)

    def finish_tasks(self):
        
        self.tasks.join()

    def end(self):
        
        self.tasks.join()
        self.stop = True
        for thread in self.workers:
            thread.join()
