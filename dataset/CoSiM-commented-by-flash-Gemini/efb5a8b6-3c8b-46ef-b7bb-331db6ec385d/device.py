


from threading import Event, Thread, Lock, Condition

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

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.new_script = Event()
        self.new_script_received = None
        self.barrier = None
        self.script_lock = Lock()
        self.lock_dict = {}

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                
                device.barrier = self.barrier
                for loc in device.sensor_data:
                    if loc not in self.lock_dict:
                        self.lock_dict[loc] = Lock()
            
            for device in devices:
                device.lock_dict = self.lock_dict

    def assign_script(self, script, location):
        
        
        
        self.script_lock.acquire()
        self.new_script_received = (script, location)
        self.new_script.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script_runners = []
        self.neighbours = []
        self.script = None
        self.location = None

        self.new_script = Event()
        self.script_lock = Lock()
        self.wait_for_data = Event()

    def run(self):
        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break

            
            self.script_runners = []
            new_scr = self.new_script
            scr_lock = self.script_lock
            wait_data = self.wait_for_data
            for _ in range(8):
                script_runner = ScriptRunner(self, new_scr, scr_lock, wait_data)
                self.script_runners.append(script_runner)
                script_runner.start()

            
            for (script, location) in self.device.scripts:
                self.script = script
                self.location = location
                self.new_script.set()
                self.wait_for_data.wait()
                self.wait_for_data.clear()

            
            while True:
                self.device.new_script.wait()
                self.device.new_script.clear()
                self.device.script_lock.release()

                self.script = self.device.new_script_received[0]
                self.location = self.device.new_script_received[1]

                if self.script is None:
                    break

                self.new_script.set()
                self.wait_for_data.wait()
                self.wait_for_data.clear()
                self.device.scripts.append((self.script, self.location))

            
            self.script = None
            self.location = None
            self.neighbours = None
            for script_runner in self.script_runners:
                self.new_script.set()
                self.wait_for_data.wait()
                self.wait_for_data.clear()

            for script_runner in self.script_runners:
                script_runner.join()

            


            self.device.barrier.wait()

class ScriptRunner(Thread):
    

    def __init__(self, device_thread, new_script, script_lock, wait_for_data):
        
        Thread.__init__(self)
        self.device_thread = device_thread
        self.new_script = new_script
        self.script_lock = script_lock
        self.wait_for_data = wait_for_data

    def run(self):
        while True:
            self.script_lock.acquire()
            self.new_script.wait()
            self.new_script.clear()
            
            script = self.device_thread.script
            location = self.device_thread.location
            neighbours = self.device_thread.neighbours
            
            self.wait_for_data.set()
            self.script_lock.release()

            if neighbours is None or location is None or script is None:
                break

            if neighbours == []:
                continue

            
            self.device_thread.device.lock_dict[location].acquire()

            script_data = []
            
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                for device in neighbours:
                    device.set_data(location, result)
                
                self.device_thread.device.set_data(location, result)

            
            self.device_thread.device.lock_dict[location].release()
