




from threading import Event, Thread, Lock, Semaphore, Lock

class ReusableBarrier(object):


    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase1()
        self.phase2()

    def phase1(self):
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:


                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()



class Device(object):
    
    
    bar1 = ReusableBarrier(1)
    event1 = Event()
    locck = []

    def __init__(self, device_id, sensor_data, supervisor):
        
        
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = []

        
        
        self.event = []
        for _ in xrange(11):
            self.event.append(Event())

        
        self.nr_threads_device = 8
        
        self.nr_thread_atribuire = 0
        
        
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1)

        
        self.thread = DeviceThread(self)
        self.thread.start()

        
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.devices = devices
        
        if self.device_id == 0:
            for _ in xrange(30):
                Device.locck.append(Lock())
            Device.bar1 = ReusableBarrier(len(devices))
            
            Device.event1.set()

    def assign_script(self, script, location):
        
        if script is not None:
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%\
            self.nr_threads_device
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
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.contor = 0

    def run(self):
        Device.event1.wait()

        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                self.device.event[self.contor].set()
                break

            
            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            self.device.event[self.contor].set()
            self.contor += 1

            
            
            self.device.bar_threads_device.wait()

            
            
            Device.bar1.wait()

class ThreadAux(Thread):
    
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device
        self.script_loc = {}
        self.contor = 0

    def run(self):
        while True:
            
            
            self.device.event[self.contor].wait()
            self.contor += 1

            
            neigh = self.device.thread.neighbours
            if neigh is None:
                break

            for script in self.script_loc:
                location = self.script_loc[script]
                
                
                Device.locck[location].acquire()
                script_data = []

                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    result = script.run(script_data)
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                
                Device.locck[location].release()

            
            self.device.bar_threads_device.wait()
