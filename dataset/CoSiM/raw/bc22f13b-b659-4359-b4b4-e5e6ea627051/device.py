


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_barrier = None
        self.location_locks = None
        self.script_lock = Lock() 
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbour_acquiring_lock = Lock() 
        self.scripts = []
        self.script_taken = [] 
        self.timepoint_done = Event()
        self.threads = []
        self.current_time_neighbours = None 
        self.other_devices = None 
        self.first_device_setup = Event() 

        self.crt_timestamp_neigh_taken = False 
        


        for i in range(8):
            self.threads.append(DeviceThread(self))
            self.threads[i].start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.other_devices = devices
        


        if self.device_id == 0:
            self.device_barrier = ReusableBarrierSem(len(devices))
            self.location_locks = []
            for _ in range(150):
                self.location_locks.append(Lock())
            self.first_device_setup.set()
        else:
        	
            for device in devices:
                if device.device_id == 0:
                    device.first_device_setup.wait()
                    self.device_barrier = device.device_barrier
                    self.location_locks = device.location_locks
                    return

    def assign_script(self, script, location):
        
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_taken.append(False)
        else:
            
            if self.device_barrier is None:
                for device in self.other_devices:
                    if device.device_id == 0:
                        self.device_barrier = device.device_barrier
                        self.location_locks = device.location_locks
                        break

            self.device_barrier.wait() 
            
            for i in range(len(self.script_taken)):
                self.script_taken[i] = False
            self.timepoint_done.set()
            self.neighbour_acquiring_lock.acquire()
            self.crt_timestamp_neigh_taken = False
            self.neighbour_acquiring_lock.release()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for thread in self.threads:
            if thread.isAlive():
                thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            self.device.neighbour_acquiring_lock.acquire()
            if self.device.crt_timestamp_neigh_taken is False:


                self.device.current_time_neighbours = self.device.supervisor.get_neighbours()
                self.device.crt_timestamp_neigh_taken = True
            self.device.neighbour_acquiring_lock.release()

            neighbours = self.device.current_time_neighbours

            if neighbours is None:
                break
            
            for (script, location) in self.device.scripts:
                self.device.script_lock.acquire()
                if self.device.script_taken[self.device.scripts.index((script, location))]:
                    self.device.script_lock.release()
                    continue
                else:
                    self.device.script_taken[self.device.scripts.index((script, location))] = True


                self.device.script_lock.release()

                
                self.device.location_locks[location].acquire()
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
                self.device.location_locks[location].release()

            self.device.timepoint_done.wait()
