




from threading import Event, Thread, Lock, Semaphore, RLock




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
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        


        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor


        self.scripts = []
        self.timepoint_done = Event()

        
        self.threads = []
        
        self.no_threads = 8
        
        
        self.timepoint_barrier = None
        
        
        self.locks = []
        
        
        self.scripts_lock = Lock()
        
        self.internal_barrier = ReusableBarrier(self.no_threads)
        
        


        self.end_timepoint = Lock()
        
        self.last_scripts = []

        
        
        if device_id == 0:
            self.init_event = Event()

        
        for thread_id in range(self.no_threads):
            thread = DeviceThread(self, thread_id)
            self.threads.append(thread)


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:
            self.timepoint_barrier = ReusableBarrier(len(devices))

            no_location = 0


            for device in devices:
                no_location += len(device.sensor_data)

            self.locks = [RLock() for _ in range(no_location)]
            self.init_event.set()
        else:
            
            for device in devices:
                if device.device_id == 0:
                    device.init_event.wait()

                    self.timepoint_barrier = device.timepoint_barrier
                    self.locks = device.locks

        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        
        
        if script is not None:
            self.scripts_lock.acquire()
            self.scripts.append((script, location))
            self.scripts_lock.release()
        else:
            self.end_timepoint.acquire()
            self.timepoint_done.set()

    def get_data(self, location):
        

        
        


        if location in self.sensor_data:
            self.locks[location].acquire()

        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        

        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        
        if self.thread_id == 0:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            
            if self.device in self.device.neighbours:
                self.device.neighbours.remove(self.device)

        while True:
            
            if self.thread_id == 0:
                self.device.scripts_lock.acquire()
                self.device.scripts += self.device.last_scripts
                self.device.last_scripts = []
                self.device.scripts_lock.release()

            self.device.internal_barrier.wait()
            
            neighbours = self.device.neighbours
            if neighbours is None:
                break

            
            while len(self.device.scripts) != 0:
                script = None

                
                self.device.scripts_lock.acquire()
                if len(self.device.scripts) != 0:

                    script, location = self.device.scripts.pop(0)
                    
                    self.device.last_scripts.append((script, location))
                self.device.scripts_lock.release()

                if script:
                    script_data = []

                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        result = script.run(script_data)



                        for device in neighbours:
                            device.set_data(location, result)

                        self.device.set_data(location, result)

		    
            self.device.timepoint_done.wait()

            
            while len(self.device.scripts) != 0:
                script = None

                
                self.device.scripts_lock.acquire()
                if len(self.device.scripts) != 0:

                    script, location = self.device.scripts.pop(0)
                    
                    self.device.last_scripts.append((script, location))
                self.device.scripts_lock.release()

                if script:
                    script_data = []

                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        result = script.run(script_data)



                        for device in neighbours:
                            device.set_data(location, result)

                        self.device.set_data(location, result)


            
            self.device.internal_barrier.wait()

            if self.thread_id == 0:
                self.device.timepoint_barrier.wait()
                

                self.device.neighbours = self.device.supervisor.get_neighbours()
                if self.device.neighbours and self.device in self.device.neighbours:
                    self.device.neighbours.remove(self.device)

                
                self.device.timepoint_done.clear()
                self.device.end_timepoint.release()
