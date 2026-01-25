


from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

from barrier import ReusableBarrierCond


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.scripts = [] 
        self.work_queue = Queue() 
        self.neighbours = [] 

        self.timepoint_done = Event() 
        self.setup_ready = Event() 
        self.neighbours_set = Event() 
        self.scripts_mutex = Semaphore(1) 
        self.location_locks_mutex = None 
        self.location_locks = {} 
        self.timepoint_barrier = None 

        self.thread = DeviceThread(self)
        
        self.workers = [DeviceWorker(self) for _ in range(8)]

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:
            self.timepoint_barrier = ReusableBarrierCond(len(devices))
            self.location_locks_mutex = Lock()
            self.setup_ready.set() 
        else:
            
            device = next(device for device in devices if device.device_id == 0)
            device.setup_ready.wait() 
            self.timepoint_barrier = device.timepoint_barrier
            self.location_locks = device.location_locks
            self.location_locks_mutex = device.location_locks_mutex

        
        with self.location_locks_mutex:
            for location in self.sensor_data.keys():
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()

        
        self.thread.start()
        for worker in self.workers:
            worker.start()


    def assign_script(self, script, location):
        
        
        self.neighbours_set.wait()

        if script is not None:
            with self.scripts_mutex:
                self.scripts.append((script, location))
            self.work_queue.put((script, location))
        else:
            self.neighbours_set.clear() 
            self.timepoint_done.set() 

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()
        for worker in self.workers:
            self.work_queue.put((None, None))

        for worker in self.workers:
            worker.join()



class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

        while True:
            


            self.device.neighbours = self.device.supervisor.get_neighbours()

            
            if self.device.neighbours is None:
                break

            for (script, location) in self.device.scripts:
                self.device.work_queue.put((script, location))

            
            
            self.device.neighbours_set.set()

            
            self.device.timepoint_done.wait()

            
            self.device.work_queue.join()

            
            self.device.timepoint_done.clear()

            
            self.device.timepoint_barrier.wait()



class DeviceWorker(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device %d Worker" % device.device_id)
        self.device = device


    def run(self):

        while True:
            
            (script, location) = self.device.work_queue.get(block=True)

            
            if script is None and location is None:
                
                self.device.work_queue.task_done()
                break

            
            
            


            with self.device.location_locks[location]:
                script_data = []

                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    result = script.run(script_data)

                    for device in self.device.neighbours:
                        device.set_data(location, result)

                    self.device.set_data(location, result)

            self.device.work_queue.task_done()
