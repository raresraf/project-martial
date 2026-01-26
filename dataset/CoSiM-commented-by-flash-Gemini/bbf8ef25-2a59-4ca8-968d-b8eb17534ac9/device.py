

from threading import Semaphore, Lock, Event, Thread
from Queue import Queue
from barrier import ReusableBarrier

class Device(object):
    
    MAX_LOCATIONS = 100
    CONST_ONE = 1

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.semaphore = []
        self.thread = DeviceThread(self)
        self.barrier = ReusableBarrier(Device.CONST_ONE)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def get_distributed_objs(self, barrier, semaphore):
        
        self.barrier = barrier
        self.semaphore = semaphore
        self.thread.start()

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            semaphore = []
            i = Device.MAX_LOCATIONS
            while i > 0:
                semaphore.append(Semaphore(value=Device.CONST_ONE))
                i = i - 1


            for device in devices:
                device.get_distributed_objs(barrier, semaphore)

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            obj = self.sensor_data[location]
        else:
            obj = None
        return obj

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    
    THREADS_TO_START = 8
    MAX_SCRIPTS = 100

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = []
        self.neighbours = []
        self.queue = Queue(maxsize=DeviceThread.MAX_SCRIPTS)

    def run(self):
        lock = Lock()
        for i in range(DeviceThread.THREADS_TO_START):
            self.threads.append(WorkerThread(self, i, self.device, self.queue, lock))
            self.threads[i].setDaemon(True)
        for thread in self.threads:
            thread.start()

        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            
            for script in self.device.scripts:
                self.queue.put(script)
            self.queue.join()

            self.device.timepoint_done.clear()

            
            
            self.device.barrier.wait()

        for thread in self.threads:
            thread.join()


class WorkerThread(Thread):
    

    def __init__(self, master, worker_id, device, queue, lock):
        
        Thread.__init__(self, name="Worker Thread %d %d" % (worker_id, device.device_id))
        self.master = master
        self.device = device
        self.queue = queue
        self.lock = lock

    def run(self):

        while True:
            
            self.lock.acquire()
            value = self.queue.empty()
            if value is False:
                (script, location) = self.queue.get()
            self.lock.release()



            if value is False:
                script_data = []

                self.device.semaphore[location].acquire()

                
                for device in self.master.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.master.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in self.master.neighbours:
                        device.set_data(location, result)
                    
                    self.master.device.set_data(location, result)

                self.device.semaphore[location].release()
                self.queue.task_done()

            if self.master.neighbours is None:
                break
