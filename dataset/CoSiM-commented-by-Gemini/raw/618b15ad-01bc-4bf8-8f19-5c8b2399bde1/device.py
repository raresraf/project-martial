


from threading import Event, Thread, Lock
from barrier import Barrier
from threadpool import ThreadPool

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.barrier = None
        self.locks = {location : Lock() for location in sensor_data}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))

            
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()
            self.script_received.set()

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
        self.thread_pool = ThreadPool(7, device)

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            for (script, location) in self.device.scripts:
                self.thread_pool.submit(neighbours, script, location)

            
            self.device.barrier.wait()

        self.thread_pool.shutdown()


from Queue import Queue
from threading import Thread

class ThreadPool(object):
    

    def __init__(self, threads_count, device):
        
        self.queue = Queue(threads_count)

        self.threads = []
        self.device = device

        
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)
            new_thread.start()

    def execute(self):
        
        while True:
            now = self.queue.get()
            if now is None:
                self.queue.task_done()
                return

            self.run_script(now)
            self.queue.task_done()

    def run_script(self, script_env_data):
        
        neighbours, script, location = script_env_data
        script_data = []

        
        for device in neighbours:
            if device.device_id != self.device.device_id:


                if location in device.sensor_data:
                    device.locks[location].acquire()
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        
        if location in self.device.sensor_data:
            self.device.locks[location].acquire()
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data)

            
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)
                    if location in device.sensor_data:


                        device.locks[location].release()

            
            self.device.set_data(location, result)
            if location in self.device.sensor_data:
                self.device.locks[location].release()

    def submit(self, neighbours, script, location):
        
        self.queue.put((neighbours, script, location))

    def shutdown(self):
        
        self.queue.join()

        for _ in self.threads:
            self.queue.put(None)

        for thread in self.threads:
            thread.join()
