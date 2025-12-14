


from threading import Event, Thread, Lock
from utils import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self, 0)

        
        self.common_barrier = None
        
        
        self.wait_initialization = Event()

        
        self.locations_locks = None
        
        self.lock_location_dict = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if not self.device_id == 0:
            
            
            self.wait_initialization.wait()
            
            self.thread.start()
        else:
            
            

            
            self.locations_locks = {}

            
            barrier_size = len(devices)
            self.common_barrier = ReusableBarrier(len(devices))

            for dev in devices:
                dev.common_barrier = self.common_barrier
                dev.locations_locks = self.locations_locks
            for dev in devices:
                if not dev.device_id == 0:
                    dev.wait_initialization.set()

            self.thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
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


class DeviceThread(Thread):
    

    def __init__(self, device, th_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.th_id = th_id

    def run(self):

        while True:
            
            self.device.common_barrier.wait()

            if self.th_id == 0:
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    break
            else:
                
                
                pass

            
            self.device.timepoint_done.wait()

            
            current_scripts = self.device.scripts

            
            for (script, location) in current_scripts:
                
                self.device.lock_location_dict.acquire()

                
                


                if not self.device.locations_locks.has_key(location):
                    self.device.locations_locks[location] = Lock()

                
                self.device.locations_locks[location].acquire()

                
                self.device.lock_location_dict.release()

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

                
                self.device.locations_locks[location].release()

            self.device.timepoint_done.clear()
from threading import Semaphore, Lock


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
                
                for i in range(self.num_threads):
                    
                    
                    threads_sem.release()
                
                count_threads[0] = self.num_threads
        
        threads_sem.acquire()
        
