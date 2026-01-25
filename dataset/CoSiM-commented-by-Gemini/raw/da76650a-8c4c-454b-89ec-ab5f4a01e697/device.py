


from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = []
        for _ in xrange(8):
            self.script_received.append(Event())
        self.scripts = []
        self.devices = None
        self.barrier = None
        self.lock = Lock()
        self.locks = {}
        self.neighbours = None

        thread_barrier = ReusableBarrier(8)
        self.threads = []
        for i in xrange(8):
            self.threads.append(DeviceThread(self, i, thread_barrier))
        for i in xrange(8):
            self.threads[i].start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        self.devices = devices
        if self.device_id == 0:
            
            barrier = ReusableBarrier(8 * len(devices))
            lock = Lock()
            for device in devices:
                device.barrier = barrier
                device.lock = lock

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            
            if location not in self.locks:
                with self.lock:
                    auxlock = Lock()
                    for device in self.devices:
                        device.locks[location] = auxlock
        else:
            for i in xrange(8):


                self.script_received[i].set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for i in xrange(8):
            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id, barrier):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id
        self.barrier = barrier

    def run(self):
        while True:
            
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            
            self.barrier.wait()
            if self.device.neighbours is None:
                break

            
            self.device.script_received[self.thread_id].wait()
            
            for i in xrange(self.thread_id, len(self.device.scripts), 8):
                (script, location) = self.device.scripts[i]
                with self.device.locks[location]:
                    script_data = []
                    
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        
                        result = script.run(script_data)

                        
                        for device in self.device.neighbours:
                            device.set_data(location, result)
                        
                        self.device.set_data(location, result)

            
            self.device.script_received[self.thread_id].clear()
            
            self.device.barrier.wait()
