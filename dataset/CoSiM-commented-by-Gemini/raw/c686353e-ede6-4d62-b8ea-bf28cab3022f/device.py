


from threading import Event, Thread, Lock
import cond_barrier


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
        self.dict_location = {}
        self.barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == 0:
            num_threads = len(devices)
            
            self.barrier = cond_barrier.ReusableBarrierCond(num_threads)
            for device in devices:
                device.barrier = self.barrier
                device.dict_location = self.dict_location

    def assign_script(self, script, location):
        
        
        if location not in self.dict_location:
            self.dict_location[location] = Lock()

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

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


class Worker(Thread):
    

    def __init__(self, worker_id, neighbours, device, dict_location):
        
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.worker_id = worker_id
        self.neighbours = neighbours
        self.device = device
        self.dict_location = dict_location
        self.scripts = []
        self.location = []

    def addwork(self, script, location):
        
        self.scripts.append(script)
        self.location.append(location)

    def run(self):
        
        i = 0


        for script in self.scripts:
            
            self.dict_location[self.location[i]].acquire()
            script_data = []
            
            for device in self.neighbours:
                data = device.get_data(self.location[i])
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location[i])
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                
                for device in self.neighbours:
                    device.set_data(self.location[i], result)
                
                self.device.set_data(self.location[i], result)
            self.dict_location[self.location[i]].release()
            i = i + 1

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            nr_worker = 0
            num_threads = 8
            workers = []

            for i in range(num_threads):
                
                
                lock_loc = self.device.dict_location
                workers.append(Worker(i, neighbours, self.device, lock_loc))

            
            for (script, location) in self.device.scripts:
                workers[nr_worker].addwork(script, location)
                nr_worker = nr_worker + 1
                if nr_worker == 8:
                    nr_worker = 0

            
            for i in range(num_threads):
                workers[i].start()
            for i in range(num_threads):
                workers[i].join()

            self.device.timepoint_done.clear()
            
            self.device.barrier.wait()
