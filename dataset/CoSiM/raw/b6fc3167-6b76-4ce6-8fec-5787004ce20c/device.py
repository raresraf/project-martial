


from threading import Thread, Lock, Condition, Event


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

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.is_root_device(devices) == 0:
            set_barriers(devices)

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def is_root_device(self, devices):
        
        is_root = 0
        for current_device in devices:
            if current_device.device_id < self.device_id:
                is_root = 1
                break
        return is_root

    def get_data(self, loc):
        
        return self.sensor_data[loc] if loc in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
        else:
            pass

    def shutdown(self):
        
        self.thread.join()

def set_barriers(devices):
    
    
    lock_set = {}
    barrier = ReusableBarrier(len(devices))
    for current_device in devices:
        current_device.barrier = barrier
        for current_location in current_device.sensor_data:
            lock_set[current_location] = Lock()
        current_device.lock_set = lock_set

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self)


        self.device = device

    def run(self):
        nr_threads = 8
        while True:
            self.device.timepoint_done.clear()
            neigh = self.device.supervisor.get_neighbours()
            self.device.barrier.wait()
            if neigh is None:
                break
            self.device.timepoint_done.wait()
            perform_s = []
            for script in self.device.scripts:
                perform_s.append(script)
            threads = []
            for i in xrange(nr_threads):
                threads.append(ExecuteSript(self.device, neigh, perform_s))
            for i in xrange(nr_threads):
                threads[i].start()
            for i in xrange(nr_threads):
                threads[i].join()
            self.device.barrier.wait()

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

    def print_barrier(self):
        "Print this barrier"
        print self.num_threads, self.count_threads

class ExecuteSript(Thread):
    


    def __init__(self, device, neighbours, perform_script):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.perform_script = perform_script

    def run(self):
        if len(self.perform_script) != 0:
            (script, location) = self.perform_script.pop()
            collected = []
            
            self.device.lock_set[location].acquire()

            
            for current_neigh in self.neighbours:
                data = current_neigh.get_data(location)
                collected.append(data)
            data = self.device.get_data(location)
            collected.append(data)

            
            if collected != []:
                result = script.run(collected)
                for current_neigh in self.neighbours:
                    current_neigh.set_data(location, result)
                self.device.set_data(location, result)

            
            self.device.lock_set[location].release()
