


from threading import Event, Thread, Lock, Condition


class Device(object):
    
    dev_barrier = None
    dev_locks = []

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.device_lock = Lock()
        self.devices = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.devices = devices

        if self.devices[0].device_id == self.device_id:
            list_loc = []
            for device_ in self.devices:
                for location in list(device_.sensor_data.viewkeys()):
                    if location not in list_loc:
                        list_loc.append(location)

            for index in range(len(list_loc)):
                Device.dev_locks.append(Lock())

            if Device.dev_barrier is None:
                Device.dev_barrier = ReusableBarrierCond(len(self.devices))

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        
        with self.device_lock:
            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        
        self.device_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.device_lock.release()

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

            self.device.script_received.wait()
            self.device.script_received.clear()

            threads_script = []

            
            
            

            for (script, location) in self.device.scripts:
                thread = ScriptThread(self.device, script, neighbours, location)
                threads_script.append(thread)

            for thread in threads_script:
                thread.start()

            for thread in threads_script:
                thread.join()

            Device.dev_barrier.wait()


class ScriptThread(Thread):
    

    def __init__(self, device, script, neighbours, location):
        
        Thread.__init__(self)


        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.location = location

    def run(self):
        script_data = []

        
        Device.dev_locks[self.location].acquire()

        
        for device_ in self.neighbours:
            data = device_.get_data(self.location)

            if data is not None:
                script_data.append(data)

        
        data = self.device.get_data(self.location)

        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)
            


            for device_ in self.neighbours:
                device_.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        
        Device.dev_locks[self.location].release()


class ReusableBarrierCond(object):
    

    def __init__(self, num_threads):
        

        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
