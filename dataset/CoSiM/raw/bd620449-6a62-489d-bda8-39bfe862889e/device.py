


from threading import *


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

        self.lock_data = Lock()
        self.lock_location = []
        self.time_barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            self.time_barrier = ReusableBarrierSem(len(devices)) 

            for device in devices:
                device.time_barrier = self.time_barrier

            loc_num = 0

            for device in devices:
                for location in device.sensor_data:
                    loc_num = max(loc_num, location) 
            for i in range(loc_num + 1):
                self.lock_location.append(Lock()) 

            for device in devices:
                device.lock_location = self.lock_location 

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        with self.lock_data:
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
            slaves = []
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() 

            
            for (script, location) in self.device.scripts:
                slave = SlaveThread(script, location, neighbours, self.device) 
                slaves.append(slave)
                slave.start()

            for i in range(len(slaves)):
                slaves.pop().join()

            self.device.time_barrier.wait() 

class SlaveThread(Thread):
    def __init__(self, script, location, neighbours, device):
        

        Thread.__init__(self, name="Slave Thread of Device %d" % device.device_id)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        
        device = self.device
        script = self.script
        location = self.location
        neighbours = self.neighbours
        
        data = device.get_data(location)
        input_data = []
        this_lock = device.lock_location[location]

        if data is not None:
            input_data.append(data) 

        with this_lock: 
            for neighbour in neighbours:
                temp = neighbour.get_data(location) 

                if temp is not None:
                    input_data.append(temp)

            if input_data != []: 
                result = script.run(input_data) 

                for neighbour in neighbours:
                    neighbour.set_data(location, result) 

                device.set_data(location, result) 


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
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire()
         
    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            self.count_threads1 = self.num_threads
         
        self.threads_sem2.acquire()
