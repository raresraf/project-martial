


from threading import Condition
from threading import Thread, Event, RLock


class ReusableBarrierCond(object):
    

    def __init__(self, num_threads):


        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class ScriptThread(Thread):
    

    def __init__(self, script, device, location, neighbours):
        Thread.__init__(self)
        self.script = script
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.locks = device.locks

    def run(self):

        script_data = []

        
        self.locks.get(self.location).acquire()



        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        data = self.device.get_data(self.location)

        if data is not None:
            script_data.append(data)

        if not script_data == []:
            
            result = self.script.run(script_data)

            for device in self.neighbours:
                device.set_data(self.location, result)

            self.device.set_data(self.location, result)

        
        self.locks.get(self.location).release()


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
        self.barrier = None
        self.locks = None
        self.creator = True

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.creator is True:
            barrier = ReusableBarrierCond(len(devices))
            locks = {}

            
            for location in self.sensor_data:
                if not locks.get(location):
                    locks.__setitem__(location, RLock())

            
            for index in xrange(0, len(devices)):
                devices[index].barrier = barrier
                devices[index].locks = locks
                devices[index].creator = False
        else:
            pass

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]

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

    def run(self):

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()
            
            self.device.timepoint_done.clear()

            
            script_threads = []

            for (script, location) in self.device.scripts:
                
                if not self.device.locks.get(location):
                    self.device.locks.__setitem__(location, RLock())
                
                script_thread = \
                    ScriptThread(script, self.device, location, neighbours)
                
                script_threads.append(script_thread)

            
            for script_thread in script_threads:
                script_thread.start()

            
            for script_thread in script_threads:
                script_thread.join()

            
            self.device.barrier.wait()
