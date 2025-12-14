


from threading import Event, Thread, RLock, Lock, Semaphore

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

class MyLock(object):
    

    def __init__(self, deviceId, zone):
        
        self.lock = RLock()
        self.dev = deviceId
        self.zone = zone

    def acquire(self):
        
        self.lock.acquire()

    def release(self):
        
        self.lock.release()

def get_leader(devices):
    
    leader = devices[0].device_id
    for dev in devices:
        if dev.device_id < leader:
            leader = dev.device_id
    return leader

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
        self.barrier = None
        self.global_lock = None
        self.gl1 = None
        self.lock_list = None


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        leader = get_leader(devices)
        if self.device_id == leader:
            global_lock = RLock()
            gl1 = RLock()
            lock_list = []
            barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                dev.barrier = barrier
                dev.global_lock = global_lock
                dev.gl1 = gl1
                dev.lock_list = lock_list

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

class MyThread(Thread):
    

    def __init__(self, device, scripts, neighbours):
        
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):

        dev = self.device
        scripts = self.scripts
        neighbours = self.neighbours

        for (script, location) in scripts:

            script_data = []

            
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)
                
                for device in neighbours:
                    device.set_data(location, result)
                dev.set_data(location, result)

def contains(my_list, searched):
    
    for elem in my_list:
        if elem == searched:
            return 1
    return 0

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def existent_lock(self, new_lock):
        
        for lock in self.device.lock_list:
            if new_lock.dev == lock.dev:
                if new_lock.zone == lock.zone:
                    return 1
        return 0

    def get_index(self, dev, zone):
        
        my_list = self.device.lock_list
        for i in range(len(my_list)):
            if dev == my_list[i].dev:
                if zone == my_list[i].zone:
                    return i
        
        
        return -1

    def run(self):

        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()
            self.device.script_received.clear()

            
            self.device.global_lock.acquire()

            
            my_list = []

            for (script, location) in self.device.scripts:

                
                new_lock = MyLock(self.device.device_id, location)
                if self.existent_lock(new_lock) == 0:
                    self.device.lock_list.append(new_lock)
                index = self.get_index(self.device.device_id, location)

                
                for device in neighbours:
                    new_lock = MyLock(device.device_id, location)
                    if self.existent_lock(new_lock) == 0:
                        self.device.lock_list.append(new_lock)
                    index = self.get_index(device.device_id, location)
                    if contains(my_list, index) == 0:
                        my_list.append(index)

            self.device.global_lock.release()

            
            self.device.gl1.acquire()
            for index in my_list:
                self.device.lock_list[index].acquire()
            self.device.gl1.release()

            
            length = len(self.device.scripts)
            if length == 1:
                trd = MyThread(self.device, self.device.scripts, neighbours)
                trd.start()
                trd.join()
            else:
                tlist = []
                for i in range(length):
                    lst = [self.device.scripts[i]]
                    trd = MyThread(self.device, lst, neighbours)
                    trd.start()
                    tlist.append(trd)
                for i in range(length):
                    tlist[i].join()

            
            for index in my_list:
                self.device.lock_list[index].release()

            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
