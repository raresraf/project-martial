/**
 * @file device.py
 * @brief Semantic documentation for device.py. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */


import Queue as q
from threading import Event, Thread, Lock
from Barrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.setup_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.data_locks = None
        self.scripts_arrived = False

    def __str__(self):
        
        return "Device %d" % self.device_id

    def assign_barrier(self, barrier):
		
		self.barrier = barrier

    def setup_devices(self, devices):
        
        self.data_locks = {loc: Lock() for loc in self.sensor_data}

        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))

            for device in devices:
                device.assign_barrier(barrier)

        self.setup_done.set()

    def assign_script(self, script, location):
        
        if script is not None:
            if location not in self.data_locks:
                self.data_locks[location] = Lock()

            self.scripts.append((script, location))
            self.script_received.set()
            self.scripts_arrived = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            self.data_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_locks[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        workers = WorkerManager(device=self.device, num_workers=8)



        while True:
            neighbours = self.device.supervisor.get_neighbours()
            
            if neighbours is None:
                break

            
            while True:

                
                if self.device.scripts_arrived or self.device.timepoint_done.wait():
                    if self.device.scripts_arrived:
                        self.device.scripts_arrived = False
                        for (script, location) in self.device.scripts:
                            workers.add_job(script, location, neighbours)
                    
                    else:
                        self.device.timepoint_done.clear()
                        self.device.scripts_arrived = True
                        break

            workers.wait_all()

            self.device.barrier.wait()

        workers.end()


class Worker(Thread):
    
    def __init__(self, device, jobs):
        super(Worker, self).__init__()
        self.device = device
        self.jobs = jobs



    def run(self):
        while True:
            script, location, neighbours = self.jobs.get()

            
            if script is None and neighbours is None:
                self.jobs.task_done()
                break

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

            self.jobs.task_done()


class WorkerManager:
    
    def __init__(self, device, num_workers):
        self.device = device
        self.jobs = q.Queue()
        self.workers = []

        for _ in range(num_workers):
            thread = Worker(device, self.jobs)
            self.workers.append(thread)

        for worker in self.workers:
            worker.start()

    def add_job(self, script, location, neighbours):
        self.jobs.put((script, location, neighbours))

    def wait_all(self):
        self.jobs.join()

    def end(self):
        for _ in self.workers:
            self.add_job(None, None, None)

        for worker in self.workers:
            worker.join()
