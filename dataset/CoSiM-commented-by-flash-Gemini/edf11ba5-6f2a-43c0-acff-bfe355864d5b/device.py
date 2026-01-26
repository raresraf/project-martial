


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        
        self.script_received = Event()

        
        self.scripts = []

        
        self.lock_locations = []

        
        self.barrier = ReusableBarrierSem(0)

        
        self.thread = DeviceThread(self)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        barrier = ReusableBarrierSem(len(devices))

        
        if self.device_id == 0:
            nr_locations = 0

            
            for i in range(len(devices)):
                for location in devices[i].sensor_data.keys():
                    if location > nr_locations:
                        nr_locations = location
            
            nr_locations += 1

            
            for i in range(nr_locations):
                lock_location = Lock()
                self.lock_locations.append(lock_location)

            for i in range(len(devices)):
                
                devices[i].barrier = barrier

                
                for j in range(nr_locations):
                    devices[i].lock_locations.append(self.lock_locations[j])

                
                devices[i].thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            
            self.scripts.append((script, location))
        else:
            
            self.script_received.set()

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


    def run(self):
        
        workers = []

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.script_received.wait()
            self.device.script_received.clear()

            
            for (script, location) in self.device.scripts:
                workers.append(Worker(self.device, script,
                                        location, neighbours))

            
            for i in range(len(workers)):
                workers[i].start()

            
            for i in range(len(workers)):
                workers[i].join()

            
            workers = []

            
            
            self.device.barrier.wait()



class Worker(Thread):
    

    def __init__(self, device, script, location, neighbours):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def solve_script(self, script, location, neighbours):
        
        
        self.device.lock_locations[location].acquire()

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

        
        self.device.lock_locations[location].release()

    def run(self):
        
        self.solve_script(self.script, self.location, self.neighbours)
