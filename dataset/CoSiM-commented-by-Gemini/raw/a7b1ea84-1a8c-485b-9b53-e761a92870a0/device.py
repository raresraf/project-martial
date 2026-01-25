


from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data


        self.supervisor = supervisor
        self.timepoint_done = Event()

        
        self.buffer = Queue()
        
        self.fresh = []

        
        self.scripts_by_location = {}

        
        self.master = DeviceMaster(self)
        self.master.start()

        
        
        self.workers = [DeviceWorker(self) for _ in xrange(8)]
        for worker in self.workers:
            worker.start()

        
        self.local_lock = {loc: Lock() for loc in self.sensor_data.keys()}

        
        
        self.barrier = None
        self.location_lock = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for i in xrange(1, len(devices)):
                devices[i].barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.fresh.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        data = None
        if location in self.sensor_data:
            self.local_lock[location].acquire()
            data = self.sensor_data[location]
        return data

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.local_lock[location].release()

    def shutdown(self):
        
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceWorker(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.master = self.device.master

    def _run_one_script(self, script, location):
        
        
        
        script_data = []

        
        for device in self.master.neighbours:
            if device != self.device:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            
            result = script.run(script_data)

            
            for device in self.master.neighbours:
                if device != self.device:
                    device.set_data(location, result)

            
            self.device.set_data(location, result)

    def _run_all_by_location(self, location):
        
        for script in self.master.scripts_by_location[location]:
            self._run_one_script(script, location)

    def run(self):
        while True:
            
            (script, location) = self.master.buffer.get()

            if location is None:
                
                self.master.buffer.task_done()
                break

            if script is None:
                self._run_all_by_location(location)
            else:
                self._run_one_script(script, location)

            
            self.master.buffer.task_done()


class DeviceMaster(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.buffer = self.device.buffer
        self.fresh = self.device.fresh
        self.scripts_by_location = self.device.scripts_by_location

    def run(self):
        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()

            
            if self.neighbours is None:
                
                for _ in self.device.workers:
                    self.buffer.put((None, None))
                break

            
            for loc in self.device.scripts_by_location.keys():
                self.buffer.put((None, loc))

            
            self.device.timepoint_done.wait()

            
            while self.fresh:
                elem = (script, location) = self.fresh.pop(0)

                if location not in self.scripts_by_location:
                    self.scripts_by_location[location] = [script]
                else:
                    self.scripts_by_location[location].append(script)

                self.buffer.put(elem)

            
            self.buffer.join()
            
            self.device.barrier.wait()
            
            self.device.timepoint_done.clear()
