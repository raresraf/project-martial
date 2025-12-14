


from threading import Event, Thread, Semaphore, RLock
from barrier import Barrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id


        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.scripts = []
        self.scripts_queue = []
        self.threads = []
        self.cores_no = 8
        self.neighbours = []

        self.timepoint_done = Event()
        self.queue_lock = RLock()
        self.location_locks = {}
        self.queue_sem = Semaphore(value=0)
        self.timepoint_barrier = Barrier()
        self.neighbours_barrier = Barrier(self.cores_no)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        for dev in devices:
            for location in dev.sensor_data.keys():
                if location not in self.location_locks:
                    self.location_locks[location] = RLock()

        
        self.timepoint_barrier.set_num_threads(len(devices)*self.cores_no)
        self.timepoint_barrier = devices[0].timepoint_barrier
        self.location_locks = devices[0].location_locks

        
        for i in xrange(self.cores_no):
            self.threads.append(DeviceThread(self, i))
            self.threads[i].start()

    def assign_script(self, script, location):
        
        
        
        if script is not None:
            with self.queue_lock:
                self.timepoint_done.clear()
                
                
                self.scripts.append((script, location))
                self.scripts_queue.append((script, location))
            self.queue_sem.release()
        else:
            with self.queue_lock:
                self.timepoint_done.set()
            
            for _ in xrange(self.cores_no):
                self.queue_sem.release()

    def recreate_queue(self):
        
        
        
        with self.queue_lock:
            for script in self.scripts:
                self.scripts_queue.append(script)
                self.queue_sem.release()

    def get_data(self, location):
        
        return self.sensor_data[location] \
               if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for i in xrange(self.cores_no):


            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device %d Thread %d" %
                        (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id


    def run_script(self, script, location):
        
        script_data = []
        
        
        with self.device.location_locks[location]:
            
            for dev in self.device.neighbours:
                data = dev.get_data(location)
                if data is not None:
                    script_data.append(data)

            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

            
            for dev in self.device.neighbours:
                dev.set_data(location, result)
                
                self.device.set_data(location, result)


    def run(self):
        while True:
            
            if self.thread_id == 0:
                
                self.device.recreate_queue()
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            self.device.neighbours_barrier.wait()

            if self.device.neighbours is None:
                break

            
            while True:
                
                self.device.queue_sem.acquire()
                self.device.queue_lock.acquire()
                if self.device.timepoint_done.is_set() and \
                    len(self.device.scripts_queue) == 0:
                    self.device.queue_lock.release()
                    break
                else:
                    (script, location) = self.device.scripts_queue.pop(0)
                self.device.queue_lock.release()

                self.run_script(script, location)
            self.device.timepoint_barrier.wait()
