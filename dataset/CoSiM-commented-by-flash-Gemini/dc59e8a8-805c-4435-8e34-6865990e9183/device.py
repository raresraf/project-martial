


from threading import Event, Thread, Semaphore, Lock


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
                for i in range(self.num_threads):
                    
                    
                    threads_sem.release()
                    
                count_threads[0] = self.num_threads
                
        threads_sem.acquire()
        


class Device(object):
    
    barrier = None

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
        
        Device.barrier = ReusableBarrier(len(devices))

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
        self.script_received.set()

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

            script_index = 0
            script_threads = []
            length_scripts_threads = 0
            while True:
                if script_index < len(self.device.scripts):
                    if length_scripts_threads < 8:
                        thread = self.call_threads(neighbours, script_index)
                        if thread.is_alive():
                            script_threads.append((thread, True))
                            length_scripts_threads += 1
                        script_index += 1
                    else:
                        local_index = 0
                        while local_index < len(script_threads):
                            if (script_threads[local_index][0].isAlive()
                                    and script_threads[local_index][1] is True):
                                script_threads[local_index][1] = False
                                length_scripts_threads -= 1
                            local_index += 1
                elif self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
                    self.device.script_received.clear()
                    break
                else:
                    self.device.script_received.wait()
                    self.device.script_received.clear()

            Device.barrier.wait()

    def call_threads(self, neighbours, index):
        
        thread = MyThread(self.device, neighbours, self.device.scripts[index])
        thread.start()
        thread.join()
        return thread




class MyThread(Thread):
    
    locations_locks = {}

    def __init__(self, device, neighbours, (script, location)):
        
        Thread.__init__(self)
        self.location, self.script = location, script


        self.device, self.neighbours = device, neighbours

        if location not in MyThread.locations_locks:
            MyThread.locations_locks[location] = Lock()

    def run(self):
        MyThread.locations_locks[self.location].acquire()
        
        script_data = []
        
        for device in self.neighbours:
            if device.get_data(self.location) is not None:
                script_data.append(device.get_data(self.location))
        
        if self.device.get_data(self.location) is not None:
            script_data.append(self.device.get_data(self.location))

        
        if script_data != []:
            
            result = self.script.run(script_data)
            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)
        MyThread.locations_locks[self.location].release()
