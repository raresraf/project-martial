


from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrierCond

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
        self.lock_data = Lock()
        self.locks_locations = {}
        self.barrier = None
        self.worker_threads_no = 8
        self.worker_threads = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        
        if self.device_id == 0:

            self.barrier = ReusableBarrierCond(len(devices))

            
            all_locations = []
            for device in devices:
                for location in device.sensor_data:
                    if location not in all_locations:
                        all_locations.append(location)
                        self.set_lock_on_location(location)

            for device in devices:
                device.set_locks_locations(self.locks_locations)
                device.set_barrier(self.barrier)

    def set_barrier(self, pbarrier):
        
        self.barrier = pbarrier


    def set_lock_on_location(self, plocation):
        
        self.locks_locations[plocation] = Lock()

    def set_locks_locations(self, plocks):
        
        self.locks_locations = plocks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def spread_scripts_to_threads(self):
        
        script_no = 0
        for (script, location) in self.device.scripts:
            thread_idx = script_no % self.device.worker_threads_no
            self.device.worker_threads[thread_idx].add_script(script, location)
            script_no += 1

    def run(self):

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            
            for i in range(self.device.worker_threads_no):
                worker_thread = DeviceWorkerThread(self, neighbours)
                self.device.worker_threads.append(worker_thread)

            
            self.spread_scripts_to_threads()

            
            for i in range(len(self.device.worker_threads)):
                self.device.worker_threads[i].start()

            
            for i in range(len(self.device.worker_threads)):
                self.device.worker_threads[i].join()

            
            del self.device.worker_threads[:]

            
            self.device.timepoint_done.clear()

            
            self.device.barrier.wait()


class DeviceWorkerThread(Thread):
    
    def __init__(self, device_thread, neighbours):
        super(DeviceWorkerThread, self).__init__()
        self.master_thread = device_thread
        self.device_neighbours = neighbours
        self.assigned_scripts = []

    def add_script(self, script, location):
        
        if script is not None:


            self.assigned_scripts.append((script, location))

    def run(self):
        
        for (script, location) in self.assigned_scripts:
            
            self.master_thread.device.locks_locations[location].acquire()

            script_data = []
            
            for device in self.device_neighbours:
                
                device.lock_data.acquire()
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
                device.lock_data.release()

            
            
            self.master_thread.device.lock_data.acquire()
            data = self.master_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)
            self.master_thread.device.lock_data.release()



            if script_data != []:
                
                result = script.run(script_data)
                
                
                for device in self.device_neighbours:
                    device.lock_data.acquire()
                    device.set_data(location, result)
                    device.lock_data.release()

                
                
                self.master_thread.device.lock_data.acquire()
                self.master_thread.device.set_data(location, result)
                self.master_thread.device.lock_data.release()

            
            self.master_thread.device.locks_locations[location].release()
