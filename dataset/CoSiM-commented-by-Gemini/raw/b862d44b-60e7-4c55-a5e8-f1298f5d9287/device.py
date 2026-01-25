


from threading import Event, Thread, Lock
from barrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)

        
        self.barrier = ReusableBarrier(0)

        
        self.locations_lock = []

        
        self.thread_list = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        barrier = ReusableBarrier(len(devices))

        
        if self.device_id == 0:

            
            locations = []
            for device in devices:
                if device is not None:
                    locations.append(max(device.sensor_data.keys()))
            no_locations = max(locations) + 1

            
            for i in xrange(no_locations):


                self.locations_lock.append(Lock())

            for device in devices:
                if device is not None:

                    
                    device.barrier = barrier

                    
                    for i in xrange(no_locations):
                        device.locations_lock.append(self.locations_lock[i])

                    
                    device.thread.start()

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
    



    def __init__(self, device, neighbours, script, location):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location

    def run(self):

        self.device.locations_lock[self.location].acquire()

        script_data = []
        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        self.device.locations_lock[self.location].release()


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

            
            
            for (script, location) in self.device.scripts:
                self.device.thread_list.append(MyThread(self.device, neighbours, script, location))

            for thread in self.device.thread_list:
                thread.start()

            for thread in self.device.thread_list:
                thread.join()

            
            self.device.thread_list = []

            
            self.device.timepoint_done.clear()

            
            self.device.barrier.wait()
