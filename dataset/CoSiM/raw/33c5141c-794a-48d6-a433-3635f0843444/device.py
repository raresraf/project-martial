


from threading import Event, Thread, Lock
from barrier import Barrier
from threadpool import ThreadPool


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        self.barrier = None
        self.locks = {}
        self.script_received = False

        for location in sensor_data:
            self.locks[location] = Lock()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.script_received = True
            self.scripts.append((location, script))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location not in self.sensor_data:
            return None

        self.locks[location].acquire()
        return self.sensor_data[location]

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.pool = ThreadPool(8)

    def run(self):
        
        
        while True:
            
            local_devices = self.device.supervisor.get_neighbours()
            if local_devices is None:
                break

            
            temp_set = set(local_devices)
            temp_set.add(self.device)
            local_devices = list(temp_set)
            

            
            while True:

                
                if self.device.script_received or self.device.timepoint_done.wait():


                    if self.device.script_received:
                        self.device.script_received = False

                        
                        for (location, script) in self.device.scripts:
                            self.pool.add_task(location, script, local_devices)
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_received = True
                        break
            
            self.pool.wait_tasks()
            self.device.barrier.wait()

        
        self.pool.join_threads()




from Queue import Queue
from threading import Thread

def execute_script(location, script, local_devices):
    

    data_collection = []

    
    for device in local_devices:
        data = device.get_data(location)
        if data is not None:
            data_collection.append(data)

    if data_collection != []:
        result = script.run(data_collection)

        
        for device in local_devices:
            device.set_data(location, result)

class ThreadPool(object):

    

    def __init__(self, threads_count):
        

        self.tasks = Queue(threads_count)
        self.threads = []

        for _ in xrange(threads_count):
            self.threads.append(Thread(target=self.run))

        for thread in self.threads:
            thread.start()


    def run(self):
        

        while True:

            location, script, local_devices = self.tasks.get()

            if script is None and local_devices is None:
                self.tasks.task_done()
                return

            execute_script(location, script, local_devices)
            self.tasks.task_done()

    def add_task(self, location, script, local_devices):
        

        self.tasks.put((location, script, local_devices))

    def wait_tasks(self):
        

        self.tasks.join()

    def join_threads(self):
        

        self.wait_tasks()

        for _ in xrange(len(self.threads)):
            self.add_task(None, None, None)

        for thread in self.threads:
            thread.join()
