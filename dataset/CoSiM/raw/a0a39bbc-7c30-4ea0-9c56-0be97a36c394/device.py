


from threading import Event, Thread, Lock, Semaphore

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.lock = {}
        self.barrier = None
        self.devices = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        self.devices = devices
        self.barrier = ReusableBarrierSem(len(self.devices))

        for location in self.sensor_data:
            self.lock[location] = Lock()
        for device in devices:
            for location in device.sensor_data:
                self.lock[location] = Lock()

        for i in xrange(len(self.devices)):
            self.devices[i].barrier = self.barrier
            self.devices[i].lock = self.lock

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


class MyThread(Thread):

    def __init__(self, my_id, device, neighbours, lock, script, location):
        Thread.__init__(self, name="Thread %d from device %d" % (my_id, device.device_id))
        self.device = device
        self.my_id = my_id
        self.neighbours = neighbours
        self.lock = lock
        self.script = script
        self.location = location

    
    def run(self):
        with self.lock[self.location]:
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

    def shutdown(self):
        self.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)

        self.device = device
        self.numThreads = 0
        self.listThreads = []

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()

            
            
            for (script, location) in self.device.scripts:
                if len(self.listThreads) < 8:
                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.append(thread)
                    thread.start()
                    self.numThreads += 1
                else:
                    index = -1
                    for i in xrange(len(self.listThreads)):
                        if not self.listThreads[i].is_alive():
                            self.listThreads[i].join()
                            index = i

                    self.listThreads.remove(self.listThreads[index])

                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.insert(index,thread)
                    self.listThreads[index].start()
                    self.numThreads += 1

            for i in xrange(len(self.listThreads)):
                self.listThreads[i].join()

            self.device.timepoint_done.wait()
            
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            
            self.device.barrier.wait()



class ReusableBarrierSem():
    

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
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()
