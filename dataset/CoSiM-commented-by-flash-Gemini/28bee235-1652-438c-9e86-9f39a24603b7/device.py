


from threading import Event, Thread
from threading import Semaphore, Lock
from Barrier import ReusableBarrier


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

        self.initialization_event = Event()
        self.free_threads = Semaphore(value=8)
        self.locations = []
        self.barrier = None

        self.device_threads = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        num_devices = len(devices)
        if self.device_id is 0:
            locations = []
            number_of_locations = 30

            while number_of_locations > 0:
                locations.append(Lock())
                number_of_locations = number_of_locations - 1

            barrier = ReusableBarrier(num_devices)

            for i in range(0, num_devices):
                devices[i].initialization_event.set()
                devices[i].locations = locations
                devices[i].barrier = barrier

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

    def clear_threads(self):
        
        for thread in self.device_threads:
            thread.join()

        self.device_threads = []

    def shutdown(self):
        
        self.clear_threads()
        self.thread.join()

def execute(device, script, location, neighbours):
    
    with device.locations[location]:
        script_data = []
        
        for dev in neighbours:
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data)

            
            for dev in neighbours:
                dev.set_data(location, result)
            
            device.set_data(location, result)
        device.free_threads.release()

class DeviceThread(Thread):
    

    def __init__(self, device):


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        
        self.device.initialization_event.wait()

        while True:

            
            

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:

                

                self.device.free_threads.acquire()
                device_thread = Thread(target=execute, \
                           args=(self.device, script, location, neighbours))

                
                
                

                device_thread.start()
                self.device.device_threads.append(device_thread)

            
            self.device.timepoint_done.clear()

            
            
            self.device.clear_threads()

            
            
            
            self.device.barrier.wait()
