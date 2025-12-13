

from threading import Event, Thread, Lock
from Queue import Queue
from reentrantbarrier import Barrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.location_locks = {} 
        self.barrier = None 
        self.ready_to_get_script = False 
        self.all_devices = None 

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        
        self.barrier = barrier

    def broadcast_barrier(self, devices):
        
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(self.barrier)


    def setup_devices(self, devices):
        
        
        self.all_devices = devices
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            self.broadcast_barrier(devices)

    def assign_script(self, script, location):
        
        if script is None:
            self.timepoint_done.set()
            return
        else:
            
            if self.location_locks.setdefault(location, None) is None:
                self.location_locks[location] = Lock()
                
                self.ready_to_get_script = True

            
            self.broadcast_lock_for_location(location)

            self.scripts.append((script, location))
            self.script_received.set()

    def broadcast_lock_for_location(self, location):
        
        for device_no in xrange(len(self.all_devices)):
            self.all_devices[device_no].location_locks[location] = self.location_locks[location]



    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data



    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_count = 8
        self.pool = Queue(self.thread_count)
        self.threads = []
        self.create_workers()
        self.start_workers()

    def create_workers(self):
        
        for _ in xrange(self.thread_count):
            self.threads.append(Thread(target=self.execute_script))

    def start_workers(self):
        
        for thread in self.threads:
            thread.start()


    def collect_data_from_neighbours(self, neighbours, location):
        
        result = []
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    result.append(data)
        return result

    def execute_script(self):
        
        
        neighbours, script, location = self.pool.get()

        while True:
            
            if neighbours is None and script is None and location is None:
                self.pool.task_done()
                break

            
            script_data = []
            self.device.location_locks[location].acquire()
            
            
            collected_data = self.collect_data_from_neighbours(neighbours, location)
            if collected_data:
                script_data = script_data + collected_data
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                for device in neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)
            
            self.device.location_locks[location].release()
            
            
            self.pool.task_done()
            neighbours, script, location = self.pool.get()

    def run(self):
        while True:
            


            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            while True:

                self.device.timepoint_done.wait()

                
                if not self.device.ready_to_get_script:
                    self.device.timepoint_done.clear()
                    self.device.ready_to_get_script = True
                    break

                
                else:
                    for (script, location) in self.device.scripts:
                        self.pool.put((neighbours, script, location))
                    self.device.ready_to_get_script = False


            
            self.pool.join()
            self.device.barrier.wait()

        
        self.pool.join()
        for _ in xrange(self.thread_count):
            self.pool.put((None, None, None))
        for thread in self.threads:
            thread.join()
        self.device.location_locks.clear()
