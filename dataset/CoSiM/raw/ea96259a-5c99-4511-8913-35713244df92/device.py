


from threading import Event, Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import barrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.device_barrier = None
        self.script_received = Event()
        self.timepoint_done = Event()

        self.scripts = []
        self.future_list = []
        self.access_locks = {}
        for location in sensor_data:
            self.access_locks[location] = Lock()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        

        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == 0:
            device_barrier = barrier.ReusableBarrierCond(len(devices))
            for device in devices:
                device.set_barrier(device_barrier)

    def set_barrier(self, device_barrier):
        

        self.device_barrier = device_barrier

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()


    def get_data(self, location):
        

        if location in self.sensor_data:
            self.access_locks[location].acquire()
            result = self.sensor_data[location]
        else:
            result = None

        return result

    def set_data(self, location, data):
        

        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.access_locks[location].release()

    def shutdown(self):
        

        self.thread.join()


class DeviceThread(Thread):
    

    def execute(self, neighbours, script, location):
        

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

    def __init__(self, device):
        

        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPoolExecutor(max_workers=8)

    def run(self):

        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            future_list = []

            
            self.device.timepoint_done.wait()


            if self.device.script_received.is_set():
                self.device.script_received.clear()

                for (script, location) in self.device.scripts:
                    future = self.thread_pool.submit(self.execute, neighbours, script, location)
                    future_list.append(future)

            
            self.device.timepoint_done.clear()
            self.device.script_received.set()

            
            
            concurrent.futures.wait(future_list)

            self.device.device_barrier.wait()

        
        self.thread_pool.shutdown()
