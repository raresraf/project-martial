


from threading import Event, Semaphore, Lock, Thread

class ReusableBarrierSem(object):
    

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


class MyThread(Thread):
    

    def __init__(self, device, location, neighbours, script):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def run(self):
        
        self.device.locks[self.location].acquire()

        script_data = []
        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        self.device.locks[self.location].release()

def get_locations(devices):
    
    no_loc = 0

    for i in xrange(len(devices)):
        maxx = int(max(devices[i].sensor_data.keys()))
        if maxx > no_loc:
            no_loc = maxx
    return no_loc + 1 

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        
        self.locks = []
        
        self.barrier = None
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        

        if self.device_id == 0:
            
            barrier = ReusableBarrierSem(len(devices))
            for i in xrange(len(devices)):
                devices[i].barrier = barrier

            
            no_loc = get_locations(devices)
            for i in xrange(no_loc):
                lock = Lock()              
                self.locks.append(lock)    

            
            for i in xrange(no_loc):
                for j in xrange(len(devices)):
                    devices[j].locks.append(self.locks[i])

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            my_threads = []

            self.device.script_received.wait()
            self.device.script_received.clear()

            
            for (script, location) in self.device.scripts:
                
                my_threads.append(MyThread(self.device, location, neighbours, script))
                my_threads[-1].start()

            
            self.device.barrier.wait()

            for i in xrange(len(my_threads)):
                my_threads[i].join()

            
            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
