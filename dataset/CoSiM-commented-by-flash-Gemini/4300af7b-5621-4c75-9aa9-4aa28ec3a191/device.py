


from threading import Event, Thread, Semaphore
from Queue import Queue
from barrier import ReusableBarrierCond


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
        self.locations = []
        self.location_locks = None
        self.barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        lock = {}
        barrier = ReusableBarrierCond(len(devices))

        if self.barrier is None:
            
            self.get_all_locations(devices)
            for location in self.locations:
                lock[location] = Semaphore(1)

            
            for device in devices:
                if device.barrier is None and device.location_locks is None:
                    device.barrier = barrier
                    device.location_locks = lock

    def get_all_locations(self, devices):
        
        for device in devices:
            for location in device.sensor_data:
                if location not in self.locations:
                    self.locations.append(location)

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


class MyThread(Thread):
    

    def __init__(self, device, tasks):
        


        Thread.__init__(self, name="MyThread %d" % device.device_id)
        self.device = device
        self.tasks = tasks

    def run(self):
        
        while True:
            
            neighbours, script, location = self.tasks.get()

            
            if neighbours is None:
                self.tasks.task_done()
                return

            
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

            
            self.tasks.task_done()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.max_threads = 8
        self.tasks = Queue(self.max_threads)

        self.thread_list = []
        for _ in range(self.max_threads):
            self.thread_list.append(MyThread(self.device, self.tasks))

    def run(self):
        for thread in self.thread_list:
            thread.start()

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                


                for _ in self.thread_list:
                    self.tasks.put((None, None, None))
                self.tasks.join()
                break

            
            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                self.tasks.put((neighbours, script, location))

            
            self.tasks.join()

            
            self.device.barrier.wait()

            
            self.device.timepoint_done.clear()

        
        for thread in self.thread_list:
            thread.join()
        self.thread_list = []
