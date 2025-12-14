

from threading import Condition, Event, RLock, Thread

class ReusableBarrier(object):
    
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def reinit(self):
        
        self.cond.acquire()
        self.num_threads -= 1
        self.cond.release()
        self.wait()

    def wait(self):
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.start = Event()
        self.scripts = []
        self.scripts_to_process = []
        self.timepoint_done = Event()
        self.nr_script_threats = 0
        self.thread = DeviceThread(self)
        self.thread.start()
        self.script_threats = []
        self.barrier_devices = None
        self.neighbours = None
        self.cors = 8
        self.lock = None
        self.lock_self = None
        self.results = {}
        self.results_lock = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        for script in self.scripts:
            self.lock.acquire()
            self.scripts_to_process.append(script)
            self.lock.release()

        if not self.lock_self:
            lock = RLock()
            for device in devices:
                device.lock_self = lock

        self.lock_self.acquire()
        if not self.lock:
            rlock = RLock()
            for device in devices:
                device.lock = rlock
        self.lock_self.release()

        self.lock_self.acquire()
        if not self.results_lock:
            results_lock = RLock()
            for device in devices:
                device.results_lock = results_lock
        self.lock_self.release()

        self.lock_self.acquire()
        if not self.barrier_devices:
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier_devices = barrier
                device.start.set()
        self.lock_self.release()



    def assign_script(self, script, location):
        

        if script is not None:
            self.lock.acquire()
            self.scripts.append((script, location))
            self.scripts_to_process.append((script, location))
            self.script_received.set()
            self.lock.release()
        else:
            self.lock.acquire()
            self.timepoint_done.set()
            self.script_received.set()
            self.lock.release()

    def get_data(self, location):
        
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else:
            data = None
        return data

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
        
    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.device.neighbours = None

    def run(self):
        
        self.device.start.wait()
        while True:
            self.device.scripts_to_process = []
            for script in self.device.scripts:
                self.device.scripts_to_process.append(script)

            
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                self.device.barrier_devices.reinit()
                break

            self.device.results = {}
            while True:
                if not self.device.timepoint_done.is_set():
                    self.device.script_received.wait()
                    self.device.script_received.clear()

                
                if len(self.device.scripts_to_process) == 0:
                    if self.device.timepoint_done.is_set():
                        break

                
                while len(self.device.scripts_to_process):
                    list_threats = []
                    self.device.script_threats = []
                    self.device.nr_script_threats = 0
                    
                    while len(self.device.scripts_to_process) and self.device.nr_script_threats < self.device.cors:
                        script, location = self.device.scripts_to_process.pop(0)
                        list_threats.append((script, location))
                        self.device.nr_script_threats += 1

                    for script, location in list_threats:
                        script_data = []
                        
                        neighbours = self.device.neighbours
                        for device in neighbours:


                            device.lock_self.acquire()
                            data = device.get_data(location)
                            device.lock_self.release()
                            if data is not None:
                                script_data.append(data)
                        
                        self.device.lock_self.acquire()
                        data = self.device.get_data(location)
                        self.device.lock_self.release()
                        if data is not None:
                            script_data.append(data)

                        thread_script_d = ScriptThread(self.device, script, location, script_data)

                        self.device.script_threats.append(thread_script_d)
                        thread_script_d.start()

                    for thread in self.device.script_threats:
                        thread.join()

            
            for location, result in self.device.results.iteritems():
                


                for device in self.device.neighbours:
                    device.lock_self.acquire()
                    device.set_data(location, result)
                    device.lock_self.release()
                
                self.device.lock_self.acquire()
                self.device.set_data(location, result)
                self.device.lock_self.release()

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            self.device.barrier_devices.wait()

class ScriptThread(Thread):
    

    def __init__(self, device, script, location, script_data):
        
        Thread.__init__(self, name="Device Script Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.script_data = script_data

    def run(self):
        
        if self.script_data != []:
            
            result = self.script.run(self.script_data)
            
            self.device.results_lock.acquire()
            self.device.results[self.location] = result
            self.device.results_lock.release()
        self.device.nr_script_threats -= 1
