


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

        
        self.devices = []

        
        self.locations = []

        
        self.setup_start = Event()

        
        self.set_lock = Lock()
        self.get_lock = Lock()

        
        self.barrier = None

        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.devices = devices
        num_devices = len(devices)

        
        barrier = ReusableBarrierSem(num_devices)

        if self.device_id == 0:

            
            for _ in range(25):
                lock = Lock()
                self.locations.append(lock)

            
            for device in devices:
                device.locations = self.locations
                device.barrier = barrier
                device.setup_start.set()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        

        
        with self.get_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        

        
        with self.set_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

        
        self.device.setup_start.wait()

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            
            if neighbours is None:
                break

            
            self.device.script_received.wait()
            self.device.script_received.clear()

            index = 0
            workers = []
            num_scripts = len(self.device.scripts)

            
            for _ in self.device.scripts:
                worker = WorkerThread(self.device, neighbours, index)
                workers.append(worker)
                index += 1

            
            if num_scripts < 8:
                for worker in workers:
                    worker.start()

                for worker in workers:
                    worker.join()
            else:
                aux = 0
                
                while True:

                    
                    if num_scripts == 0:
                        break

                    
                    if num_scripts >= 8:
                        start = aux


                        end = aux + 8

                        for i in range(start, end):
                            workers[i].start()

                        for i in range(start, end):
                            workers[i].join()

                        aux += 8
                        num_scripts -= 8

                    
                    elif num_scripts < 8:
                        start = aux


                        end = aux + num_scripts

                        for i in range(start, end):
                            workers[i].start()

                        for i in range(start, end):
                            workers[i].join()
                        break

            
            self.device.barrier.wait()


class WorkerThread(Thread):
    

    def __init__(self, device, neighbours, index):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.index = index

    def run(self):
        
        (script, location) = self.device.scripts[self.index]

        
        self.device.locations[location].acquire()

        
        script_data = []

        
        for neighbour in self.neighbours:
            data_neigh = neighbour.get_data(location)

            if data_neigh is not None:
                script_data.append(data_neigh)

        
        own_data = self.device.get_data(location)
        if own_data is not None:
            script_data.append(own_data)

        
        if script_data:
            result = script.run(script_data)

            
            for neighbour in self.neighbours:
                neighbour.set_data(location, result)

                
                self.device.set_data(location, result)

        
        self.device.locations[location].release()

