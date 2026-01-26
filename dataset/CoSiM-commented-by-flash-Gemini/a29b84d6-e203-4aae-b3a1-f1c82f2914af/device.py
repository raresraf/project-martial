


from threading import Event, Thread, Lock
import barrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        
        self.devices = None
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        shared_barrier = barrier.ReusableBarrier(len(devices))

        
        if self.device_id == 0:
            for i in xrange(len(devices)):
                devices[i].thread.barrier = shared_barrier

        
        
        self.devices = devices

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            if location not in self.thread.locations_lock:
                loc_lock = Lock()
                for i in xrange(len(self.devices)):
                    self.devices[i].thread.locations_lock[location] = loc_lock

        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] \
        if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = None
        self.script_threads = []
        self.locations_lock = {}

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            
            self.device.timepoint_done.wait()

            count = 0
            
            for (script, location) in self.device.scripts:
                if count == 8:
                    count = 0
                    for i in xrange(len(self.script_threads)):
                        self.script_threads[i].join()
                    del self.script_threads[:]

                script_thread = ScriptThread(self.device, script, location,\
                    neighbours, count, self.locations_lock)

                self.script_threads.append(script_thread)
                script_thread.start()
                count = count + 1

            
            for i in xrange(len(self.script_threads)):
                self.script_threads[i].join()

            
            self.device.timepoint_done.clear()
            
            
            self.barrier.wait()

class ScriptThread(Thread):
    

    def __init__(self, device, script, location, neighbours, i, locations_lock):
        Thread.__init__(self, name="Script Thread %d%d" % (device.device_id, i))
        self.device = device


        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.locations_lock = locations_lock


    def run(self):
        script_data = []

        
        
        
        self.locations_lock[self.location].acquire()

        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:

            
            result = self.script.run(script_data)



            for device in self.neighbours:
                device.set_data(self.location, result)

            self.device.set_data(self.location, result)

        
        self.locations_lock[self.location].release()
