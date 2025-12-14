


from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.lock = Lock()
        self.threads = [DeviceThread(self)]
        for thread in self.threads:
            thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        


        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                device.barrier = barrier

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for thread in self.threads:
            thread.join()

class ScriptWorker(Thread):
    

    def __init__(self, data):
        

        Thread.__init__(self)
        self.device = data['device']
        self.script = data['script']
        self.location = data['location']
        self.neighbours = data['neighbours']
        self.barrier = data['barrier']

    def run(self):
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
                
                
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()

            
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()

        self.barrier.wait()


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

            no_scripts = len(self.device.scripts)

            worker_barrier = ReusableBarrierCond(no_scripts + 1)
            workers = []
            
            for (script, location) in self.device.scripts:
                workers.append(
                ScriptWorker(
                {
                'device' : self.device,
                'script' : script,
                'location' : location,
                'neighbours' : neighbours,
                'barrier' : worker_barrier
                }
                ))

            
            for worker in workers:
                worker.start()

            worker_barrier.wait()

            
            for worker in workers:
                worker.join()

            self.device.timepoint_done.clear()
            self.device.barrier.wait()
