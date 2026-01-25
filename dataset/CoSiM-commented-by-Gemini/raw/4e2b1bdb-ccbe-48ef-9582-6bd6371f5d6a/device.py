


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):


    
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        
        with self.count_lock:
            count_threads[0] -= 1
            
            if count_threads[0] == 0:
                num_threads = self.num_threads
                
                while num_threads > 0:
                    threads_sem.release()
                    num_threads = num_threads - 1
                
                count_threads[0] = self.num_threads
        
        threads_sem.acquire()
        

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.locks = []
        self.barrier = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        global_barrier = ReusableBarrier(len(devices))
        self.barrier = global_barrier
        locations = []

        
        for device in devices:
            device.barrier = global_barrier
            
            for data in device.sensor_data:
                if data not in locations:
                    locations.append(data)

        
        for location in locations:
            lock = Lock()


            self.locks.append((location, lock))

        
        for device in devices:
            device.locks = self.locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
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

class ScriptThread(Thread):
    
    def __init__(self, device, neighbours, scripts):
        
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.scripts = scripts

    def run(self):
        
        
            
        locks = dict(self.device.locks)
        for (script, location) in self.scripts:
            
            with locks[location]:
                script_data = []
                
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)
                    
                    for device in self.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        
            
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.script_received.wait()

            
            divided_scripts = [[], [], [], [], [], [], [], []]
            index = 0
            for (script, location) in self.device.scripts:
                divided_scripts[index].append((script, location))
                if index == 8:
                    index = 0
                else:
                    index = index + 1

            
            threads = []
            for s_list in divided_scripts:
                if len(s_list) != 0:
                    thread = ScriptThread(self.device, neighbours, s_list)
                    threads.append(thread)
                    thread.start()
            for thread in threads:
                thread.join()

            self.device.timepoint_done.wait()
            
            self.device.barrier.wait()
            
            self.device.script_received.clear()
