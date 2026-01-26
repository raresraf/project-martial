


from threading import Event, Thread, Lock, Semaphore
from cond_barrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.scripts_to_run = []
        self.timepoint_done = Event()
        self.setup_done = Event()
        self.thread = DeviceThread(self)

        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        nr_devices = len(devices)
        self.barrier = ReusableBarrier(nr_devices)
        self.lock_get_neigh = Lock()
        self.lock_location = {}
        self.lock_check_loc = Lock()
        self.lock_scripts = Lock()

        
        self.barrier = devices[0].barrier
        self.lock_get_neigh = devices[0].lock_get_neigh
        self.lock_location = devices[0].lock_location
        self.lock_check_loc = devices[0].lock_check_loc

        for location in self.sensor_data:
            if not self.lock_location.has_key(location):


                self.lock_location[location] = Lock()

        self.setup_done.set()

    def assign_script(self, script, location):
        
        if script is not None:
            self.lock_scripts.acquire()
            self.scripts.append((script, location))
            self.scripts_to_run.append((script, location))
            self.lock_scripts.release()
        else:
            self.lock_scripts.acquire()
            self.timepoint_done.set()
            self.lock_scripts.release()

    def get_data(self, location):
        
        return (self.sensor_data[location]
                if location in self.sensor_data else None)

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.sem_threads = Semaphore(8) 

    def run(self):
        self.device.setup_done.wait()

        while True:
            threads = []

            


            self.device.lock_get_neigh.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.lock_get_neigh.release()

            if neighbours is None:
                break

            
            self.device.lock_scripts.acquire()
            self.device.scripts_to_run = self.device.scripts[:]
            finished = (self.device.timepoint_done.is_set() and
                        len(self.device.scripts_to_run) == 0)
            self.device.lock_scripts.release()

            while not finished:
                self.device.lock_scripts.acquire()
                local_scripts_to_run = self.device.scripts_to_run[:]
                self.device.lock_scripts.release()

                for (script, location) in local_scripts_to_run:
                    
                    self.sem_threads.acquire()

                    
                    self.device.lock_check_loc.acquire()

                    if self.device.lock_location[location].locked():
                        self.device.lock_check_loc.release()
                        self.sem_threads.release()
                        continue

                    self.device.lock_location[location].acquire()

                    self.device.lock_scripts.acquire()
                    self.device.scripts_to_run.remove((script, location))
                    self.device.lock_scripts.release()

                    self.device.lock_check_loc.release()

                    thread = Thread(target=run_script, args=(self, neighbours,
                                                             script, location))
                    threads.append(thread)
                    thread.start()

                self.device.lock_scripts.acquire()
                finished = (self.device.timepoint_done.is_set() and
                            len(self.device.scripts_to_run) == 0)
                self.device.lock_scripts.release()

            
            for thread in threads:
                thread.join()

            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

def run_script(parent_device_thread, neighbours, script, location):
    
    script_data = []
    
    for device in neighbours:
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)
    
    data = parent_device_thread.device.get_data(location)
    if data is not None:
        script_data.append(data)

    if script_data != []:
        
        result = script.run(script_data)

        
        for device in neighbours:
            device.set_data(location, result)
        
        parent_device_thread.device.set_data(location, result)

    parent_device_thread.device.lock_location[location].release()
    parent_device_thread.sem_threads.release()
