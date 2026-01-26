


from threading import Event, Thread, Semaphore, current_thread, RLock


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Semaphore(0)
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.all_devices = None
        self.timepoint_sem = Semaphore(0)
        self.list_thread = []
        self.loc_lock = {}

    def __str__(self):
        
        return "[%.35s]    Device %d:" % (current_thread(),self.device_id)

    def sync_on_timepoint(self):
        for i in range(len(self.all_devices)-1):
            self.timepoint_sem.acquire()

    def setup_devices(self, devices):
        
        self.all_devices = devices

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.loc_lock[location] = RLock()
        else:
            self.timepoint_done.set()
        self.script_received.release()

    def lock(self, location):
        if not location in self.loc_lock:
            self.loc_lock[location] = RLock()
        self.loc_lock[location].acquire()

    def unlock(self, location):
        try:
            self.loc_lock[location].release()
        except:
            pass


    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        

        if location in self.sensor_data:
            self.sensor_data[location] = data


    def shutdown(self):
        
        self.thread.join()

class ScriptThread(Thread):
    
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)


        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        script_data = []
        
        if self.neighbours:
            for device in self.neighbours:
                device.lock(self.location)
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)
            
            if self.neighbours:
                for device in self.neighbours:
                    device.set_data(self.location, result)

            self.device.lock(self.location)
            self.device.set_data(self.location, result)
            self.device.unlock(self.location)

        if self.neighbours:
            for device in self.neighbours:
                device.unlock(self.location)



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
            self.device.script_received.acquire()

            for (script, location) in self.device.scripts:
                thread = ScriptThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)
                thread.start()

            for t in self.device.list_thread:
                t.join()
            self.device.list_thread = []

            for d in self.device.all_devices:
                if d == self.device:
                    continue
                d.timepoint_sem.release()

            self.device.timepoint_done.clear()
            self.device.sync_on_timepoint()
