


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
from worker import Worker


class Device(object):
    
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []


        self.devices = []
        self.cores = 8
        self.barrier = None
        self.shared_locks = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        
        self.barrier = barrier

    def set_locks(self, locks):
        
        self.shared_locks = locks

    def setup_devices(self, devices):
        
        self.devices = devices
        
        
        if self.device_id == 0:
            lbarrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                dev.set_barrier(lbarrier)

        
        max_loc = max(self.sensor_data.keys(), key=int)
        
        
        
        if  max_loc+1 > len(self.shared_locks):
            llocks = []
            for _ in range(max_loc+1):
                llocks.append(Lock())
            self.set_locks(llocks)
            for dev in self.devices:
                dev.set_locks(llocks)

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
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def distribute_scripts(self, scripts):
        
        worker_scripts = []
        for _ in range(self.device.cores):
            worker_scripts.append([])
        i = 0
        for script in scripts:
            worker_scripts[i % self.device.cores].append(script)
            i = i + 1
        return worker_scripts

    def run(self):

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.timepoint_done.wait()

            
            inner_workers = []
            
            worker_scripts = self.distribute_scripts(self.device.scripts)
            for worker_scr in worker_scripts:
                inner_thread = Worker(worker_scr,\
                                      neighbours,\
                                      self.device)
                inner_workers.append(inner_thread)
                inner_thread.start()

            for thr in inner_workers:
                thr.join()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()



from threading import Thread

class Worker(Thread):
    
    def __init__(self, script_loc, neighbours, device):
        
        Thread.__init__(self)
        self.script_loc = script_loc
        self.neighbours = neighbours
        self.script_data = []
        self.device = device

    def run(self):
        for (script, location) in self.script_loc:
            
            self.device.shared_locks[location].acquire()
            self.script_data = []
            
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    self.script_data.append(data)

            data = self.device.get_data(location)

            if data is not None:
                self.script_data.append(data)
            
            if self.script_data != []:
                result = script.run(self.script_data)
                
                for dev in self.neighbours:
                    dev.set_data(location, result)
                
                self.device.set_data(location, result)
             
            self.device.shared_locks[location].release()
