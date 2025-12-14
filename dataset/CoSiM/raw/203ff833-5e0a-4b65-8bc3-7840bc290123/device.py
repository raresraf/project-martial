


from threading import Thread, Lock, Event, Condition, Semaphore

class ReusableBarrier():
    
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
        self.timepoint_done = Event()
        self.setup_event = Event()

        self.lock_location = []
        self.lock_n = Lock()
        self.barrier = None

        self.thread_script = []
        self.num_thread = 0
        self.sem = Semaphore(value=8)

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            for _ in xrange(25):
                self.lock_location.append(Lock())

            for dev in devices:
                dev.barrier = barrier
                dev.lock_location = self.lock_location
                dev.setup_event.set()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in \
            self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

    def shutdown_script(self):
        
        for i in xrange(self.num_thread):
            self.thread_script[i].join()

        for i in xrange(self.num_thread):
            del self.thread_script[-1]

        self.num_thread = 0

class NewThreadScript(Thread):
    
    def __init__(self, parent, neighbours, location, script):
        Thread.__init__(self)


        self.neighbours = neighbours
        self.parent = parent
        self.location = location
        self.script = script

    def run(self):
        with self.parent.lock_location[self.location]:
            script_data = []
            
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.parent.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = self.script.run(script_data)

                
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.parent.set_data(self.location, result)
            self.parent.sem.release()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

        
        self.device.setup_event.wait()



        while True:
            
            with self.device.lock_n:
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    break

            
            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                self.device.sem.acquire()
                self.device.thread_script.append(NewThreadScript \
                    (self.device, neighbours, location, script))

                self.device.num_thread = self.device.num_thread + 1
                self.device.thread_script[-1].start()



            
            self.device.barrier.wait()
            
            self.device.shutdown_script()
            
            self.device.timepoint_done.clear()
            
            self.device.barrier.wait()
