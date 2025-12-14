


from threading import Event, Thread, Lock
from Queue import Queue
from barrier import ReusableBarrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        

        self.thread_number = 8
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []


        self.timepoint_done = Event()
        self.location_locks = {}
        self.devices_barrier = None
        self.setup_devices_done = Event()
        self.neighbours = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id is 0:
            self.devices_barrier = ReusableBarrier(len(devices))
            self.location_locks["master_lock"] = Lock()
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.devices_barrier = self.devices_barrier
                    dev.location_locks = self.location_locks
                dev.setup_devices_done.set()

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
        self.neighbours = []
        self.script_queue = Queue()
        self.worker_pool = []
        for _ in range(self.device.thread_number):
            self.worker_pool.append(WorkerThread(self))

    def run(self):
        

        for worker in self.worker_pool:
            worker.start()

        while True:
            


            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break

            self.device.setup_devices_done.wait()
            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                self.script_queue.put((script, location))

            self.script_queue.join()
            self.device.devices_barrier.wait()
            self.device.timepoint_done.clear()

        for _ in range(len(self.worker_pool)):
            self.script_queue.put(None)

        for worker in self.worker_pool:
            worker.join()

class WorkerThread(Thread):
    

    def __init__(self, device_thread):
        Thread.__init__(self)
        self.device_thread = device_thread

    def run(self):
        while True:
            script_pair = self.device_thread.script_queue.get()

            if script_pair is None:
                break

            script, location = script_pair

            with self.device_thread.device.location_locks["master_lock"]:
                if location not in self.device_thread.device.location_locks:
                    self.device_thread.device.location_locks[location] = Lock()

            self.device_thread.device.location_locks[location].acquire()

            script_data = []
            
            for device in self.device_thread.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)
            if script_data != []:
                
                result = script.run(script_data)

                
                for device in self.device_thread.neighbours:
                    device.set_data(location, result)
                
                self.device_thread.device.set_data(location, result)

            self.device_thread.script_queue.task_done()
            self.device_thread.device.location_locks[location].release()
