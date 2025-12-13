


from threading import Event, Thread, Semaphore, Lock
import Queue

class Device(object):
    
    
    num_threads = 8
    
    set_barrier = Event()
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.aval_data = {}
        self.wait_threads = ReusableBarrier(Device.num_threads)
        self.neighbours = None
        self.thread = []
        self.queues = [Queue.Queue() for i in range(Device.num_threads)]
        self.crt_que = 0
        for loc in sensor_data:
            self.aval_data[loc] = Event()
            self.aval_data[loc].set()
        self.lock = Lock()
        if device_id == 0:
            self.data_lock = {}
            self.wait_devices = None
            self.wholesomebarrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        for dev in devices:
            if dev.device_id == 0:
                info_dev = dev
                break
        if self.device_id == 0:
            info_dev.wait_devices = ReusableBarrier(len(devices))
            info_dev.wholesomebarrier = ReusableBarrier(len(devices) * Device.num_threads)
            Device.set_barrier.set()
        for dev in devices:
            for data in dev.sensor_data:
                if data not in info_dev.data_lock:
                    info_dev.data_lock[data] = Semaphore(1)
        Device.set_barrier.wait()
        for i in range(0, Device.num_threads):
            self.thread.append(DeviceThread(self, i, info_dev.data_lock,\
                 self.queues[i], info_dev.wait_devices, info_dev.wholesomebarrier))
            self.thread[i].start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.queues[self.crt_que].put((script, location))
            self.crt_que += 1
            self.crt_que %= Device.num_threads
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for i in self.thread:
            i.join()

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
        
        self.count_lock.acquire()
        count_threads[0] -= 1
        if count_threads[0] == 0:
            for i in range(self.num_threads):
                threads_sem.release()
            count_threads[0] = self.num_threads
        self.count_lock.release()
        threads_sem.acquire()

class DeviceThread(Thread):
    

    def __init__(self, device, id_thread, sem_loc, queue, wait_devices, wholesomebarrier):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.id_thread = id_thread
        self.device = device
        self.script_data = []
        self.aux = None
        self.script = None
        self.location = None
        self.sem_loc = sem_loc
        self.data = 0
        self.result = 0
        self.queue = queue
        self.wait_devices = wait_devices
        self.wholesomebarrier = wholesomebarrier

    def run(self):
        while True:
            
            if self.id_thread == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            for i in range(self.id_thread, len(self.device.scripts), self.device.num_threads):
                self.queue.put(self.device.scripts[i])
            
            self.device.wait_threads.wait()
            if self.device.neighbours is None:
                break
            
            while (not self.queue.empty()) or (not self.device.timepoint_done.isSet()):
                self.script_data = []
                try:
                    
                    self.aux = self.queue.get(False)
                    self.script = self.aux[0]
                    self.location = self.aux[1]
                except Queue.Empty:
                    continue
                self.sem_loc[self.location].acquire()
                


                for device in self.device.neighbours:
                    self.data = device.get_data(self.location)
                    if self.data is not None:
                        self.script_data.append(self.data)
                
                self.data = self.device.get_data(self.location)
                if self.data is not None:
                    self.script_data.append(self.data)

                if self.script_data != []:
                    
                    self.result = self.script.run(self.script_data)
                    
                    for device in self.device.neighbours:
                        if device == self.device:
                            continue


                        if self.location in device.aval_data:
                            device.set_data(self.location, self.result)
                    self.device.set_data(self.location, self.result)
                self.sem_loc[self.location].release()

            
            self.wholesomebarrier.wait()
            if self.id_thread == 0:
                
                self.device.timepoint_done.clear()
