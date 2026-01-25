


from threading import Event, Thread, Lock
from utility import ReusableBarrierCond, ThreadCollection

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        
        self.barrier = None

        
        self.locks = {}

        
        self.setup = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        for location in self.sensor_data:
            self.locks[location] = Lock()

        
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))

            
            for device in devices:
                device.barrier = barrier

            
            for device in devices:
                device.setup.set()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
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

        
        self.worker_threads = ThreadCollection(self.device, 8)

    def run(self):
        
        self.device.setup.wait()

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            
            
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                self.worker_threads.add_task(script, location, neighbours)

            
            self.device.timepoint_done.clear()

            
            self.worker_threads.queue.join()

            
            self.device.barrier.wait()

        
        self.worker_threads.end_workers()


from Queue import Queue
from threading import Condition, Thread

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


class ThreadCollection(object):
    

    def __init__(self, device, num_threads):
        
        self.device = device
        self.threads = []

        
        self.queue = Queue(num_threads)

        
        self.create_workers(num_threads)

        
        self.start_workers()

    def __str__(self):
        
        return "Thread collection belonging to device %d" % self.device.device_id

    def create_workers(self, num_threads):
        
        
        for _ in xrange(num_threads):
            new_thread = Thread(target=self.run_tasks)
            self.threads.append(new_thread)

    def start_workers(self):
        
        for thread in self.threads:
            thread.start()

    def run_tasks(self):
        
        while True:
            
            (neighbours, script, location) = self.queue.get()

            
            
            if location is None and neighbours is None and script is None:
                self.queue.task_done()
                break

            


            self.run_script(script, location, neighbours)
            self.queue.task_done()

    def run_script(self, script, location, neighbours):
        
        
        script_data = []

        
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
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

            
            self.device.set_data(location, result)

    def add_task(self, script, location, neighbours):
        


        self.queue.put((neighbours, script, location))

    def end_workers(self):
        
        
        self.queue.join()

        
        for _ in xrange(len(self.threads)):
            self.add_task(None, None, None)

        for thread in self.threads:
            thread.join()
