

from threading import Event, Thread, Lock
import ReusableBarrier


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
        self.lock = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        barrier = ReusableBarrier.ReusableBarrier(len(devices))
        lock = []
        for _ in range(0, 100):
            newlock = Lock()
            lock.append(newlock)

        for dev in devices:
            dev.barrier = barrier
            dev.lock = lock


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        

        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        self.sensor_data[location] = data


    def shutdown(self):
        
        self.thread.join()

class MyThread(Thread):
    
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours
    def run(self):
        
        self.device.lock[self.location].acquire()
        script_data = []
        
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
            
        self.device.lock[self.location].release()

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
            self.device.timepoint_done.set()

            
            threads = []
            for (script, location) in self.device.scripts:
                thread_aux = MyThread(self.device, location, script, neighbours)


                threads.append(thread_aux)
            for auxiliar_thread in threads:
                auxiliar_thread.start()
            for auxiliar_thread in threads:
                auxiliar_thread.join()

            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()


