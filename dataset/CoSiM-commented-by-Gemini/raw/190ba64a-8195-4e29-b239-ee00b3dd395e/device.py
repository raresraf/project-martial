


from threading import Event, Thread, Lock, Condition

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
        self.locations = []
        self.sync_data_lock = Lock()
        self.sync_location_lock = {}
        self.cores = 8
        self.barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            locations_number = self.get_locations_number(devices)
            for location in range(locations_number):
                self.sync_location_lock[location] = Lock()
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                device.barrier = self.barrier
                device.sync_location_lock = self.sync_location_lock

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
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

    def get_locations_number(self, devices):
        
        for device in devices:
            for location in device.sensor_data:
                if location not in self.locations:
                    self.locations.append(location)
        return len(self.locations)


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

        while True:
            

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.timepoint_done.wait()
            
            my_threads = []
            num_threads = self.device.cores
            index = 0
            if len(my_threads) == 0:
                for i in range(num_threads):
                    thread = MyThread(self)
                    my_threads.append(thread)
            
            for (script, location) in self.device.scripts:
                my_threads[index%num_threads].assign_script(script, location)
                index = index + 1
            
            for i in range(num_threads):
                my_threads[i].set_neighbours(neighbours)
                my_threads[i].start()
            for i in range(num_threads):
                my_threads[i].join()
            
            self.device.timepoint_done.clear()
            
            self.device.barrier.wait()

class MyThread(Thread):
    
    def __init__(self, parent_device_thread):
        
        Thread.__init__(self)
        self.parent = parent_device_thread
        self.scripts = []
        self.neighbours = []

    def set_neighbours(self, neighbours):
        
        self.neighbours = neighbours

    def assign_script(self, script, location):
        


        self.scripts.append((script, location))

    def run(self):
        
        for (script, location) in self.scripts:
            
            self.parent.device.sync_location_lock[location].acquire()
            script_data = []
            
            for device in self.neighbours:
                device.sync_data_lock.acquire()
                data = device.get_data(location)
                device.sync_data_lock.release()
                if data is not None:
                    script_data.append(data)
            
            self.parent.device.sync_data_lock.acquire()
            data = self.parent.device.get_data(location)
            self.parent.device.sync_data_lock.release()
            if data is not None:
                script_data.append(data)



            if script_data != []:
                
                result = script.run(script_data)
                
                for device in self.neighbours:
                    device.sync_data_lock.acquire()
                    device.set_data(location, result)
                    device.sync_data_lock.release()

                
                self.parent.device.sync_data_lock.acquire()
                self.parent.device.set_data(location, result)
                self.parent.device.sync_data_lock.release()
            self.parent.device.sync_location_lock[location].release()


class ReusableBarrierCond(object):
    
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
