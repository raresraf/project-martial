


from threading import Event, Thread, Condition, Lock
from Queue import Queue, Empty

class ReusableBarrier(object):
    
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

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbours = None
        self.num_threads = 24
        self.script_received = Event()
        self.scripts = []
        self.workers = []
        self.queue = None
        self.work_done = None
        self.barrier = None
        self.lock = None
        self.location_lock = []
        self.thread = DeviceThread(self)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == devices[0].device_id:
            self.queue = Queue()
            self.work_done = Event()
            self.barrier = ReusableBarrier(len(devices))
            self.lock = Lock()

            
            for loc in range(100):
                self.location_lock.append(Lock())

            for thread_id in range(self.num_threads):
                self.workers.append(WorkerThread(self.queue, self.work_done))
                self.workers[thread_id].start()
        else:
            self.queue = devices[0].queue
            self.work_done = devices[0].work_done
            self.barrier = devices[0].barrier
            self.lock = devices[0].lock
            self.location_lock = devices[0].location_lock

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
        if self.workers != []:
            for thread_id in range(self.num_threads):
                self.workers[thread_id].join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            self.device.lock.acquire()
            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.lock.release()

            
            if self.device.neighbours is None:
                self.device.work_done.set()
                break

            
            self.device.script_received.wait()
            self.device.script_received.clear()

            
            for (script, location) in self.device.scripts:
                self.device.queue.put((self.device, script, location))

            


            self.device.queue.join()
            
            self.device.barrier.wait()

class WorkerThread(Thread):
    

    def __init__(self, queue, job_done):
        
        Thread.__init__(self, name="Worker Thread")
        self.tasks = queue
        self.job_done = job_done

    def run(self):
        
        while True:
            try:
                (device, script, location) = self.tasks.get(False)
            except Empty:
                if self.job_done.is_set():
                    break
                else:
                    continue

            script_data = []
            
            device.location_lock[location].acquire()

            
            for neighbour in device.neighbours:
                data = neighbour.get_data(location)
                if data is not None:
                    script_data.append(data)

            
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                device.set_data(location, result)
                for neighbour in device.neighbours:
                    neighbour.set_data(location, result)

            
            device.location_lock[location].release()
            
            self.tasks.task_done()
