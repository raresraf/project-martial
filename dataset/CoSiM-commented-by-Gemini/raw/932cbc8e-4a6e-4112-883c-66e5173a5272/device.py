


from threading import Event, Thread, Lock
from Queue import Queue
from barrier import ReusableBarrierSem

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.threads = []
        self.barrier = None
        self.timepoint_done = Event()
        self.thread_queue = Queue()
        self.locks = {}
        self.thread = DeviceThread(self)
        self.thread.start()


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        for _ in range(8):
            thread = Worker(self)
            thread.start()
            self.threads.append(thread)
        
        for device in devices:
            if device is not None:
                self.devices.append(device)
        
        if self.barrier is None:
            self.barrier = ReusableBarrierSem(len(self.devices))
        
        for device in self.devices:
            if device is not None:
                if device.barrier is None:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            
            if location is not None:
                if not self.locks.has_key(location):
                    self.locks[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

        for device in self.devices:
            if not device.locks.has_key(location):
                if self.locks.has_key(location):
                    device.locks[location] = self.locks[location]

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class Worker(Thread):
    



    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

    def run(self):
        while True:
        	
            script_loc_neigh = self.device.thread_queue.get()
            if script_loc_neigh[0] is None:
                if script_loc_neigh[1] is None:
                    if script_loc_neigh[2] is None:
                        self.device.thread_queue.task_done()
                        break
            script_data = []
            
            self.device.locks[script_loc_neigh[1]].acquire()
            


            for device in script_loc_neigh[2]:
                data = device.get_data(script_loc_neigh[1])
                if data is not None:
                    script_data.append(data)

            
            data = self.device.get_data(script_loc_neigh[1])
            if data is not None:
                script_data.append(data)
            if script_data != []:
                
                result = script_loc_neigh[0].run(script_data)
                
                for device in script_loc_neigh[2]:
                    device.set_data(script_loc_neigh[1], result)
                self.device.set_data(script_loc_neigh[1], result)

            self.device.locks[script_loc_neigh[1]].release()
            self.device.thread_queue.task_done()


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
            self.device.timepoint_done.clear()
            
            for (script, location) in self.device.scripts:
                self.device.thread_queue.put((script, location, neighbours))
            self.device.thread_queue.join()
            
            self.device.barrier.wait()
        
        for _ in range(len(self.device.threads)):
            self.device.thread_queue.put((None, None, None))
        for thread in self.device.threads:
            thread.join()
