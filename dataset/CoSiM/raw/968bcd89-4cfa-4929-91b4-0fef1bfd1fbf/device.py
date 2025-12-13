


from threading import Event, Thread, Lock
from barrier import Barrier
from thread_pool import ThreadPool

class Device(object):
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.num_threads = 8
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.location_lock = {}
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            if self.barrier is None:
                self.barrier = Barrier(len(devices))

            for device in devices:
                for location in device.sensor_data:
                    if location not in self.location_lock:
                        self.location_lock[location] = Lock()
                device.barrier = self.barrier
                device.location_lock = self.location_lock

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


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(device, self.device.num_threads)

    def run(self):
        
        self.thread_pool.start_threads()

        
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            for script, location in self.device.scripts:
                self.thread_pool.queue.put((script, location, neighbours))

            
            
            self.thread_pool.queue.join()
            self.device.barrier.wait()
            self.device.timepoint_done.clear()

        
        for _ in range(self.device.num_threads):
            self.thread_pool.queue.put((None, None, None))

        self.thread_pool.end_threads()


from Queue import Queue
from threading import Thread

class ThreadPool(object):
    
    def __init__(self, device, num_threads):
        
        self.queue = Queue(num_threads)
        self.device = device
        self.threads = []
        self.num_threads = num_threads

    def start_threads(self):
        
        for _ in range(self.num_threads):
            self.threads.append(Thread(target=self.run))



        for thread in self.threads:
            thread.start()

    def run(self):
        
        while True:
            script, location, neighbours = self.queue.get()

            if script is None and location is None:
                return

            self.run_script(script, location, neighbours)
            self.queue.task_done()

    def run_script(self, script, location, neighbours):
        
        with self.device.location_lock[location]:
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

    def end_threads(self):
        
        for thread in self.threads:
            thread.join()
