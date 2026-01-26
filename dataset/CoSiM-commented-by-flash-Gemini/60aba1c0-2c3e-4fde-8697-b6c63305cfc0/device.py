


from threading import Event, Thread, Lock
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
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
        for location in sensor_data:
            self.locks[location] = Lock()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def acquire_lock(self, location):
        
        if location in self.sensor_data:
            self.locks[location].acquire()

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def release_lock(self, location):
        
        if location in self.sensor_data:
            self.locks[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=8)

    def run_script(self, script, location, neighbours):
        

        script_data = []

        
        for device in neighbours:
            if device.device_id != self.device.device_id:

                
                device.acquire_lock(location)

                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        
        self.device.acquire_lock(location)

        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data)

            
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

                    
                    device.release_lock(location)

            
            self.device.set_data(location, result)

            
            self.device.release_lock(location)

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            futures = []

            


            while True:

                
                if self.device.script_received.isSet() or self.device.timepoint_done.wait():
                    if self.device.script_received.isSet():
                        self.device.script_received.clear()

                        
                        
                        for (script, location) in self.device.scripts:
                            futures.append(self.executor.submit(self.run_script, script,
                                                                location, neighbours))

                    else:
                        wait(futures, timeout=None, return_when=ALL_COMPLETED)
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break

            
            self.device.barrier.wait()

        
        self.executor.shutdown()
