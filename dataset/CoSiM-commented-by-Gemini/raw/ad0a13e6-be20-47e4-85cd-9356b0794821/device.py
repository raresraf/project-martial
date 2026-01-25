


from threading import Event, Thread, Lock, Condition

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
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.locks_location = []
        self.barrier_timepoint = None


        self.thread = DeviceThread(self)
        self.thread.start()



    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            
            iteration = 0
            while iteration < 100:
                iteration += 1
                lock = Lock()
                self.locks_location.append(lock)

            for device in devices:
                device.locks_location = self.locks_location
                device.barrier_timepoint = barrier


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class Worker(Thread):
    
    def __init__(self, thread_id, device, neighbors, nr_scripts, scripts):
        Thread.__init__(self)
        self.neighbors = neighbors
        self.device = device
        self.scripts = scripts
        self.nr_scripts = nr_scripts
        self.thread_id = thread_id

    def run(self):
        for index in range(self.thread_id, self.nr_scripts, 8):


            (script, location) = self.scripts[index]

            with self.device.locks_location[location]:
                script_data = []
                
                for device in self.neighbors:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    


                    result = script.run(script_data)

                    
                    for device in self.neighbors:
                        device.set_data(location, result)

                    self.device.set_data(location, result)


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.list_workers = []

    def run(self):

        while True:

            neighbors = self.device.supervisor.get_neighbours()
            if neighbors is None:
                break

            self.device.timepoint_done.wait()

            
            nr_scripts = len(self.device.scripts)
            
            for thread_id in range(0, 8):

                worker = Worker(thread_id, self.device, neighbors, nr_scripts, self.device.scripts)


                self.list_workers.append(worker)
                worker.start()

            
            for worker in self.list_workers:
                worker.join()

            self.device.timepoint_done.clear()
            
            self.device.barrier_timepoint.wait()


