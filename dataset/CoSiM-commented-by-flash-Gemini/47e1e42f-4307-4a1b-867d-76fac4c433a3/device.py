


from threading import Event, Thread, Lock

from reusable_barrier import ReusableBarrier
from thread_pool import ThreadPool

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.barrier = None

        self.locks = {}
        for location in sensor_data:
            self.locks[location] = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        nr_devices = len(devices)
        if self.device_id == 0:
            self.barrier = ReusableBarrier(nr_devices)
            
            for device in devices:
                if device.device_id:
                    device.barrier = self.barrier
        

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            self.locks[location].acquire()

        return self.sensor_data[location] if location in self.sensor_data else None


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

        self.thread_pool = ThreadPool(8, self.device)

    def run(self):
        
        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    neighbours.remove(device)
                    break

            for (script, location) in self.device.scripts:
                self.thread_pool.submit_task(neighbours, script, location)

            
            self.device.barrier.wait()

        
        self.thread_pool.join_threads()


from threading import Thread
from Queue import Queue

class ThreadPool(object):
    

    def __init__(self, threads_count, device):
        

        self.threads = []
        self.device = device
        self.threads_count = threads_count
        self.queue = Queue(threads_count)

        self.create_threads(threads_count)

    def create_threads(self, threads_count):
        
        i = 0
        while i < threads_count:
            thread = Thread(target=self.execute_task)
            self.threads.append(thread)
            i += 1

        for thread in self.threads:
            thread.start()

    def submit_task(self, neighbours, script, location):
        

        self.queue.put((neighbours, script, location))

    def execute_task(self):
        

        while True:
            
            elem = self.queue.get()
            neighbours = elem[0]
            script = elem[1]
            location = elem[2]

            if neighbours is None and script is None and location is None:
                return

            self.run_script(neighbours, script, location)

    def run_script(self, neighbours, script, location):
        

        script_data = []

        
        for device in neighbours:
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

    def join_threads(self):
        

        i = 0
        while i < self.threads_count:
            self.submit_task(None, None, None)
            i += 1

        for thread in self.threads:
            thread.join()
