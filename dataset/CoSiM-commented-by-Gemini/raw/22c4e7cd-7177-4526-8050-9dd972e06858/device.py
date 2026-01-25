


from threading import Event, Thread, Lock, Condition


from threading import Semaphore, Lock


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
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.general_barrier = None
        self.lock_till_init = Semaphore()
        self.lock_for_certain_place = None
        self.dictionary_lock = Condition(Lock())
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self, 0)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            
            

            
            self.lock_for_certain_place = {}

            
            self.general_barrier = ReusableBarrier(len(devices))

            for i in range(len(devices)):
                devices[i].general_barrier = self.general_barrier
                devices[i].lock_for_certain_place = self.lock_for_certain_place
            for i in range(len(devices)):
                if not devices[i].device_id == 0:
                    devices[i].lock_till_init.acquire()


        elif not self.device_id == 0:
            self.lock_till_init.release()
        self.thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    
    def __init__(self, device, th_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            self.device.general_barrier.wait()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            


            self.device.timepoint_done.wait()
            
            current_scripts = self.device.scripts
            
            for (script, location) in current_scripts:
                
                self.device.dictionary_lock.acquire()

                
                if not location in self.device.lock_for_certain_place:
                    self.device.lock_for_certain_place[location] = Condition(Lock())
                self.device.lock_for_certain_place[location].acquire()
                
                self.device.dictionary_lock.release()
                script_data = []
                
                for i in range(len(neighbours)):
                    if neighbours[i].get_data(location) is not None:
                        script_data.append(neighbours[i].get_data(location))
                        i = i + 1
                
                if self.device.get_data(location) is not None:
                    script_data.append(self.device.get_data(location))

                if script_data != []:
                    
                    result = script.run(script_data)
                    
                    
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                
                self.device.lock_for_certain_place[location].release()
            self.device.timepoint_done.clear()
