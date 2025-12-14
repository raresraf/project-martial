


from threading import Event, Thread, RLock, Semaphore
from barrier import ReusableBarrierSem

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor, max_workers=8):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.scripts = []
        self.master = Thread(target=self.master_func)


        self.master.start()
        self.active_workers = Semaphore(max_workers)

        
        self.root_device = None

        
        self.step_barrier = None
        self.data_locks = {}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        for dev in devices:
            if dev.device_id == 0:
                self.root_device = dev

        if self.device_id == 0:
            
            
            
            self.step_barrier = ReusableBarrierSem(len(devices))

            
            for device in devices:
                for (location, _) in device.sensor_data.iteritems():


                    if location not in self.data_locks:
                        self.data_locks[location] = RLock()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set()

    def get_data(self, location):
        
        with self.root_device.data_locks[location]:
            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        
        with self.root_device.data_locks[location]:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.master.join()

    def master_func(self):
        
        while True:
            
            neighbours = self.supervisor.get_neighbours()
            if neighbours is None:
                break



            self.scripts_received.wait()

            workers = []
            
            for (script, location) in self.scripts:
                
                self.active_workers.acquire()

                
                worker = Thread(target=self.worker_func, \
                    args=(script, location, neighbours))
                workers.append(worker)
                worker.start()

            
            for worker in workers:
                worker.join()

            
            self.scripts_received.clear()
            
            self.root_device.step_barrier.wait()


    def worker_func(self, script, location, neighbours):
        
        with self.root_device.data_locks[location]:
            
            script_data = []
            
            for dev in neighbours:
                data = dev.get_data(location)
                if data is not None:
                    script_data.append(data)

            
            data = self.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                for dev in neighbours:
                    dev.set_data(location, result)

                
                self.set_data(location, result)

        
        self.active_workers.release()
