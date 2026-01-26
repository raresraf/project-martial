


from threading import Event, Thread, Lock, Condition
import multiprocessing


class ReusableBarrier(object):
    
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

        self.timepoint_done = Event()
        self.got_neighbours = Event()

        self.neighbours = []
        self.thread_list = []

        self.device_barrier = None
        self.data_lock = None

        self.counter = 0

        self.nr_thread = multiprocessing.cpu_count()
        barrier = ReusableBarrier(self.nr_thread)
        lock = Lock()
        for i in xrange(self.nr_thread):
            thread = DeviceThread(self, i, barrier, lock)


            self.thread_list.append(thread)
            thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        for device in devices:
            if self.device_id > device.device_id:
                return None
        self.device_barrier = ReusableBarrier(len(devices))
        self.data_lock = dict()
        for device in devices:
            for location in device.sensor_data:
                if location not in self.data_lock:
                    self.data_lock[location] = Lock()
        for device in devices:
            device.set_barrier(self.device_barrier)
            device.set_data_lock(self.data_lock)

    def set_barrier(self, device_barrier):
        
        self.device_barrier = device_barrier

    def set_data_lock(self, data_lock):
        
        self.data_lock = data_lock

    def acquire_lock(self, location):
        
        self.data_lock[location].acquire()

    def release_lock(self, location):
        
        self.data_lock[location].release()

    def assign_script(self, script, location):
        
        if script is not None:
            
            pos = self.counter
            self.thread_list[pos].script_list.append((script, location))
            
            self.thread_list[self.counter].script_received.set()
            self.counter = (self.counter + 1) % self.nr_thread
        else:
            
            self.timepoint_done.set()
            for thread in self.thread_list:


                thread.script_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        
        self.sensor_data[location] = data

    def shutdown(self):
        
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id, thread_barrier, thread_lock):
        
        Thread.__init__(self, name="D:%d T:%d" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id
        self.script_list = []
        self.thread_barrier = thread_barrier
        self.thread_lock = thread_lock
        self.script_received = Event()

    def run_scripts(self, index, neighbours):
        
        size = len(self.script_list)
        while index < size:
            
            (script, location) = self.script_list[index]
            script_data = []
            
            
            self.device.acquire_lock(location)
            
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
            
            
            self.device.release_lock(location)
            index += 1
        return index

    def run(self):
        while True:
            if self.thread_id == 0:
                
                self.device.neighbours = \
                    self.device.supervisor.get_neighbours()
                self.device.got_neighbours.set()
            else:
                self.device.got_neighbours.wait()
            neighbours = self.device.neighbours

            if neighbours is None:
                
                break

            
            index = 0
            while not self.device.timepoint_done.is_set():
                index = self.run_scripts(index, neighbours)
                
                self.script_received.wait()
                self.script_received.clear()
            
            self.run_scripts(index, neighbours)

            


            self.thread_barrier.wait()
            
            if self.thread_id == 0:
                self.device.timepoint_done.clear()
                self.device.got_neighbours.clear()
                self.device.device_barrier.wait()
            self.thread_barrier.wait()
