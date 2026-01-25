


from threading import Event, Thread, Lock, Condition
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
        self.scripts = []
        self.neighbours = []

        self.barrier = None
        self.locks = []
        self.timepoint_done = Event()
        self.tasks_ready = Event()
        self.tasks = Queue()
        self.simulation_ended = False

        
        self.master = DeviceThreadMaster(self)
        self.master.start()

        
        self.workers = []
        for i in xrange(8):
            worker = DeviceThreadWorker(self, i)
            self.workers.append(worker)
            worker.start()

    def __str__(self):
        
        return "Device [%d]" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            locks = [Lock() for _ in xrange(24)]
            for device in devices:
                device.barrier = barrier
                device.locks = locks

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
        
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceThreadMaster(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device [%d] Thread Master" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            self.device.neighbours = self.device.supervisor.get_neighbours()

            
            if self.device.neighbours is None:
                
                self.device.simulation_ended = True
                self.device.tasks_ready.set()
                
                break

            
            self.device.timepoint_done.wait()

            
            for task in self.device.scripts:
                self.device.tasks.put(task)

            
            self.device.tasks_ready.set()

            
            self.device.tasks.join()

            
            self.device.tasks_ready.clear()


            self.device.timepoint_done.clear()

            
            self.device.barrier.wait()


class DeviceThreadWorker(Thread):
    

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device [%d] Thread Worker [%d]" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        
        while not self.device.simulation_ended:
            
            self.device.tasks_ready.wait()

            try:
                
                script, location = self.device.tasks.get(block=False)

                
                self.device.locks[location].acquire()

                script_data = []

                
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                
                if len(script_data) > 0:
                    
                    result = script.run(script_data)

                    
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                
                self.device.locks[location].release()

                
                self.device.tasks.task_done()
            except Empty:
                pass
