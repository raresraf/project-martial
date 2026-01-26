


from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = None
        self.locations_locks = None
        self.lock = None
        self.devices = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        

        root = max(devices)
        if self == root:
            map_locks = {}
            lock = Lock()
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.set_barrier(barrier)
                device.set_locations_locks(map_locks)
                device.set_lock(lock)
        self.devices = devices

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

    def set_barrier(self, barrier):
        
        self.thread = DeviceThread(self, barrier)
        self.thread.start()

    def set_locations_locks(self, locations_locks):
        
        self.locations_locks = locations_locks

    def set_lock(self, lock):
        
        self.lock = lock

    def acquire_location(self, location):
        
        location = str(location)
        self.lock.acquire()
        if (location in self.locations_locks) is False:
            self.locations_locks[location] = Lock()
        self.locations_locks[location].acquire()
        self.lock.release()

    def release_location(self, location):
        
        location = str(location)


        self.locations_locks[location].release()


class DeviceThread(Thread):
    

    def __init__(self, device, barrier):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
    def run(self):

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.barrier.wait()
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            threads = []

            
            for (script, location) in self.device.scripts:

                script_data = []

                self.device.acquire_location(location)

                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    thread = ScriptThread(script, script_data, self.device, neighbours, location)
                    thread.start()
                    threads.append(thread)
                else:
                    self.device.release_location(location)

            for thread in threads:
                thread.join()

class ScriptThread(Thread):
    
    def __init__(self, script, data, device, neighbours, location):
        
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.script = script
        self.data = data
        self.device = device
        self.neighbours = neighbours
        self.location = location
    def run(self):
        result = self.script.run(self.data)

        
        for device in self.neighbours:
            device.set_data(self.location, result)

        
        self.device.set_data(self.location, result)
        self.device.release_location(self.location)

