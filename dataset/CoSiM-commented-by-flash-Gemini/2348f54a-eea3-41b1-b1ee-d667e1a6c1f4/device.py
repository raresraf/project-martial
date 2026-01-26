


from threading import Event, Thread, Lock
import barrier

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.data_lock = Lock()
        self.list_locks = {}
        self.barrier = None
        self.devices = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.devices = devices

        
        
        if self.device_id == self.devices[0].device_id:
            self.barrier = barrier.ReusableBarrierCond(len(self.devices))
            for dev in self.devices:
                for location in dev.sensor_data:
                    self.list_locks[location] = Lock()
        else:
            
            self.barrier = devices[0].get_barrier()
            self.list_locks = devices[0].get_list_locks()
        
        self.thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set()


    def get_barrier(self):
        
        return self.barrier

    def get_list_locks(self):
        
        return self.list_locks

    def get_data(self, location):
        
        with self.data_lock:
            if location in self.sensor_data:
                data = self.sensor_data[location]
            else:
                data = None
        return data

    def set_data(self, location, data):
        
        with self.data_lock:
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

            
            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            
            threads = []
            for (script, location) in self.device.scripts:
                threads.append(
                    ScriptThread(self.device, script, location, neighbours))
                if len(threads) == 8:
                    for thr in threads:
                        thr.start()
                    for thr in threads:
                        thr.join()
                    threads = []
            
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()

            
            self.device.barrier.wait()



class ScriptThread(Thread):
    

    def __init__(self, device, script, location, neighbours):
        
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        
        
        self.device.list_locks[self.location].acquire()

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

        
        self.device.list_locks[self.location].release()
