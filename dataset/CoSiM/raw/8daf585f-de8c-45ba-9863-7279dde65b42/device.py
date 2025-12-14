


import cond_barrier
from threading import Event, Thread, Lock


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.threads = []

        self.neighbourhood = None
        self.map_locks = {}
        self.threads_barrier = None
        self.barrier = None
        self.counter = 0
        self.threads_lock = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            num_threads = len(devices)
            
            self.barrier = cond_barrier.ReusableBarrier(num_threads * 8)

            for device in devices:
                device.barrier = self.barrier
                device.map_locks = self.map_locks

        
        self.threads_barrier = cond_barrier.ReusableBarrier(8)
        for i in range(8):
            self.threads.append(DeviceThread(self, i, self.threads_barrier))

        
        for thread in self.threads:
            thread.start()


    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

        
        if location not in self.map_locks:


            self.map_locks[location] = Lock()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        

        
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, id, barrier):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id = id
        self.thread_barrier = barrier

    def run(self):
        while True:
            
            if self.id == 0:
                
                self.device.neighbourhood = self.device.supervisor.get_neighbours()

            
            self.thread_barrier.wait()

            if self.device.neighbourhood is None:
                break 

            
            self.device.timepoint_done.wait()

            
            while True:
                
                with self.device.threads_lock:
                    if self.device.counter == len(self.device.scripts):
                        break
                    (script, location) = self.device.scripts[self.device.counter]
                    self.device.counter = self.device.counter + 1
                
                self.device.map_locks[location].acquire()
                script_data = []

                for device in self.device.neighbourhood:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    for device in self.device.neighbourhood:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                self.device.map_locks[location].release()

            
            self.device.barrier.wait()
            if self.id == 0:
                
                
                self.device.counter = 0
                self.device.timepoint_done.clear()
