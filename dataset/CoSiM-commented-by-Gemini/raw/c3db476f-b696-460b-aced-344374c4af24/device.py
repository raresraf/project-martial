


from threading import Lock, Thread, Event, Condition

max_size = 100

class MyThread(Thread):
    
    def __init__(self, dev_thread, neighbors, location, script):
        Thread.__init__(self)
        self.dev_thread = dev_thread
        self.neighbors = neighbors
        self.location = location
        self.script = script

    def run(self):
        self.dev_thread.device.location_lock[self.location].acquire()
        script_data = []
        
        for device in self.neighbors:


            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.dev_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            
            for device in self.neighbors:
                device.set_data(self.location, result)
            
            self.dev_thread.device.set_data(self.location, result)
        self.dev_thread.device.location_lock[self.location].release()

class ReusableBarrier():
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
        self.cond_barrier = None
        self.location_lock = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == devices[0].device_id:
            self.cond_barrier = ReusableBarrier(len(devices))
            self.location_lock = []
            for i in range(0, max_size):
                self.location_lock.append(Lock())
            for dev in devices:
                dev.cond_barrier = self.cond_barrier
                dev.location_lock = self.location_lock


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


            self.device.script_received.clear()

            thread_list = []
            for (script, location) in self.device.scripts:
                thread_list.append(MyThread(self, neighbours, location, script))

            for thr in thread_list:
                thr.start()

            for thr in thread_list:
                thr.join()

            
            self.device.cond_barrier.wait()
            self.device.timepoint_done.wait()
