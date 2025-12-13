


from Queue import Queue
from threading import Thread

class Worker(Thread):
    
    def __init__(self, tasks, device):
        
        Thread.__init__(self)
        self.tasks = tasks
        self.device = device
        self.daemon = True
        self.start()

    def run(self):
        
        while True:
            neighbours, script, location = self.tasks.get()



            if neighbours is None:
                self.tasks.task_done()
                break
            with self.device.locations_locks[location]:
                self._script(neighbours, script, location)
            self.tasks.task_done()

    def _script(self, neighbours, script, location):
        
        script_data = []
        for neighbour in neighbours:
            data = neighbour.get_data(location)
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)



        if script_data != []:
            result = script.run(script_data)

            for neighbour in neighbours:
                neighbour.set_data(location, result)
            self.device.set_data(location, result)


class ThreadPool(object):
    
    def __init__(self, num_threads):
        
        self.tasks = Queue(num_threads)
        self.threads = []
        self.device = None

    def set_device(self, device, num_threads):
        
        self.device = device
        for _ in range(num_threads):
            self.threads.append(Worker(self.tasks, self.device))

    def add_tasks(self, neighbours, location, script):
        
        self.tasks.put((neighbours, location, script))

    def wait_completion(self):
        
        self.tasks.join()

    def end_threads(self):
        
        self.tasks.join()
        for _ in range(len(self.threads)):
            self.add_tasks(None, None, None)

        for thread in self.threads:
            thread.join()





from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
from ThreadPool import ThreadPool

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.barrier = None
        self.locations_locks = {}
        self.devices = []
        for location in sensor_data:
            self.locations_locks[location] = Lock()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        barrier = ReusableBarrierSem(len(devices))
        self.barrier = barrier
        for dev in devices:
            dev.barrier = barrier
            dev.locations_locks = self.locations_locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.locations_locks[location] = Lock()
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
        self.thread_pool = ThreadPool(8)

    def run(self):
        
        self.thread_pool.set_device(self.device, 8)

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.device.script_received .wait()
            for (script, location) in self.device.scripts:
                self.thread_pool.add_tasks(neighbours, script, location)

            self.device.script_received .clear()
            self.thread_pool.wait_completion()
            self.device.barrier.wait()

        self.thread_pool.end_threads()
