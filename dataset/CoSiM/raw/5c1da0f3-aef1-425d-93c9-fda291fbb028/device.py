




import threading
from threading import Thread
from Queue import Queue
from cond_barrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.location_locks = {loc: threading.Lock() for loc in self.sensor_data}
        self.supervisor = supervisor
        self.scripts = []

        self.thread = DeviceThread(self)
        self.thread.start()

        
        self.scripts_queue = Queue()
        
        self.workers_queue = Queue()

        
        self.barrier = None


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = barrier

    def assign_script(self, script, location):
        
        self.scripts_queue.put((script, location))

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def get_data_synchronize(self, location):
        
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def set_data_synchronize(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.stop = False

    def run(self):
        
        num_workers = 16
        
        workers = []
        
        workers_queue = Queue()

        
        for i in range(num_workers):
            workers.append(WorkerThread(self.device, i, workers_queue))
        for worker in workers:
            worker.start()

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            neighbours = [x for x in neighbours if x != self.device]
            for worker in workers:
                worker.neighbours = neighbours

            
            for script in self.device.scripts:
                workers_queue.put(script)

            
            while True:
                script, location = self.device.scripts_queue.get()
                if script is None:
                    break
                
                self.device.scripts.append((script, location))
                workers_queue.put((script, location))

            
            workers_queue.join()
            
            self.device.barrier.wait()

        
        for worker in workers:
            workers_queue.put((None, None))
        for worker in workers:
            worker.join()


class WorkerThread(Thread):
    

    def __init__(self, device, worker_id, queue):
        
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.device = device
        self.neighbours = []
        self.worker_id = worker_id
        self.queue = queue

    def run(self):
        while True:
            script, location = self.queue.get()
            if script is None:
                self.queue.task_done()
                break

            
            script_data = []
            
            for device in self.neighbours:
                data = device.get_data_synchronize(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data_synchronize(location)
            if data is not None:
                script_data.append(data)

            
            if script_data != []:
                
                result = script.run(script_data)

                
                
                for device in self.neighbours:
                    device.set_data_synchronize(location, result)
                
                self.device.set_data_synchronize(location, result)
            self.queue.task_done()
