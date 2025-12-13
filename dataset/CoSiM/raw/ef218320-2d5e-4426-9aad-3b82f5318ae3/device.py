


from threading import Event, Thread, RLock, Lock, Semaphore, Condition


class Device(object):
    
    barrier = None

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.run_script = RLock()
        self.scripts_sem = Semaphore(8)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if Device.barrier == None and self.device_id == 0:
            Device.barrier = ReusableBarrier(len(devices))

    def assign_script(self, script, location):
        
        self.run_script.acquire()
        self.script_received.set()
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
        self.run_script.release()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        
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

            self.device.script_received.wait()

            
            self.device.run_script.acquire()
            dictionar = {}
            i = 0
            for (script, location) in self.device.scripts:
                self.device.scripts_sem.acquire()
                thread = MyThread(self.device, neighbours, location, script)
                dictionar[i] = thread
                dictionar[i].start()
                i = i + 1
            self.device.run_script.release()
            for idx in range(0, len(dictionar)):
                dictionar[idx].join()

            Device.barrier.wait()
            self.device.timepoint_done.wait()



class MyThread(Thread):
    
    lockForLocations = {}

    def __init__(self, device, neighbours, location, script):
        
        Thread.__init__(self)
        self.location = location
        self.script = script
        self.device = device


        self.neighbours = neighbours

        if location not in MyThread.lockForLocations:
            MyThread.lockForLocations[location] = Lock()

    def run(self):
        MyThread.lockForLocations[self.location].acquire()
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
        MyThread.lockForLocations[self.location].release()
        self.device.scripts_sem.release()

class ReusableBarrier(object):
    
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
