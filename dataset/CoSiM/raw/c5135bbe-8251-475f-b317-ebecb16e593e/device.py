


from threading import Event, Thread, Lock
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

        self.barrier = None
        self.locks = {}

        self.thread = DeviceThread(self)
        self.thread.start()


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        for loc in self.sensor_data:
            self.locks[loc] = Lock()

        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))

            
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

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
        self.threads = []
        self.queue = Queue()
        self.create_threads()

    def create_threads(self):
        
        for _ in xrange(8):


            thread = Thread(target=self.work)
            self.threads.append(thread)

        for thread in self.threads:
            thread.start()

    def join_threads(self):
        
        
        for _ in xrange(8):
            self.queue.put((None, None, None))

        
        self.queue.join()

        for thread in self.threads:
            thread.join()

    def work(self):
        
        while True:
            script, location, neighbours = self.queue.get()

            
            if script is None:
                self.queue.task_done()
                break

            script_data = []

            
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

            data = self.device.get_data(location)

            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)

                for device in neighbours:


                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)

                self.device.set_data(location, result)

            self.queue.task_done()

    def run(self):


        while True:
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break

            while True:

                
                if self.device.script_received.isSet():


                    self.device.script_received.clear()

                    for (script, location) in self.device.scripts:
                        self.queue.put((script, location, neighbours))

                
                if self.device.timepoint_done.isSet():
                    self.device.timepoint_done.clear()
                    self.device.script_received.set()
                    break

            
            self.queue.join()

            
            self.device.barrier.wait()

        self.join_threads()
