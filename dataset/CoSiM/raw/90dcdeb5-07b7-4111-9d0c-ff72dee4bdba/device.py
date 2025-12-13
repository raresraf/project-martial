


from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond
from thread_pool import ThreadPool


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
        self.devices_synchronized = Event()
        self.location_semaphores = {}
        self.scripts_lock = Lock()
        self.new_location_lock = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            
            barrier = ReusableBarrierCond(len(devices))
            location_semaphores = {}
            new_location_lock = Lock()
            
            for device in devices:
                device.barrier = barrier
                device.location_semaphores = location_semaphores
                device.new_location_lock = new_location_lock
                device.devices_synchronized.set()
        
        self.devices_synchronized.wait()


    def assign_script(self, script, location):
        

        
        
        self.new_location_lock.acquire()
        if location not in self.location_semaphores:
            self.location_semaphores[location] = Semaphore()
        self.new_location_lock.release()

        
        


        self.scripts_lock.acquire()
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
        self.script_received.set()
        self.scripts_lock.release()


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
        self.executor = ThreadPool(8)

    def run(self):
        
        self.executor.start_workers()
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                
                self.executor.shutdown()
                break

            
            script_done = {}

            
            self.device.scripts_lock.acquire()
            
            while not self.device.timepoint_done.isSet() or self.device.script_received.isSet():
                
                self.device.scripts_lock.release()
                
                self.device.script_received.wait()
                self.device.script_received.clear()

                
                for (script, location) in self.device.scripts:
                    if (script, location) in script_done:
                        continue

                    
                    self.executor.submit((self.device, neighbours, script, location))
                    script_done[(script, location)] = True
                
                self.device.scripts_lock.acquire()
            
            self.device.scripts_lock.release()

            
            self.executor.wait_all()
            
            self.device.barrier.wait()
            
            self.device.timepoint_done.clear()


from Queue import Queue
from threading import Thread

class ThreadPool(object):
    

    def __init__(self, num_threads):
        
        self._queue = Queue()
        self._num_threads = num_threads
        self._workers = []
        self._init_workers()

    def _init_workers(self):
        
        for _ in xrange(self._num_threads):
            self._workers.append(WorkerThread(self._queue))

    def submit(self, args):
        
        self._queue.put(args)

    def wait_all(self):
        
        self._queue.join()

    def start_workers(self):
        
        for worker in self._workers:
            worker.start()

    def shutdown(self):
        
        for _ in xrange(self._num_threads):
            self._queue.put(None)
        for worker in self._workers:
            worker.join()


class WorkerThread(Thread):
    

    def __init__(self, queue):
        
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            try:
                
                current_device, neighbours, script, location = self.queue.get()
            except TypeError:
                
                self.queue.task_done()
                break
            else:
                
                current_device.location_semaphores[location].acquire()

                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = current_device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    current_device.set_data(location, result)

                
                current_device.location_semaphores[location].release()
                
                self.queue.task_done()
