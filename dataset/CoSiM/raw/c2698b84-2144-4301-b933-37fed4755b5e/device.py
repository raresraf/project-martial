


from threading import Event, Thread, Condition

class ReusableBarrierCond():

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
        self.devices = None
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()



    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        self.devices = devices
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(self.devices))
            for device in devices:


                device.barrier = self.barrier


    def assign_script(self, script, location):
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class ScriptsThread(Thread):
    

    def __init__(self, device, scripts, neighbours):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours


    def run(self):
        for (script, location) in self.scripts:
            script_data = []
            
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    if data != self.device.get_data(location):
                        script_data.append(data)
                
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                for device in self.neighbours:
                    if result > device.get_data(result):
                        device.set_data(location, result)
                    
                if result > self.device.get_data(result):


                    self.device.set_data(location, result)
            
        self.device.thread.barrier.wait()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = None
        self.list_of_threads = []


    def run(self):
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.timepoint_done.wait()
            now_thread = 0
            now_script = 0
            
            for script in self.device.scripts:
                if now_script == 8:
                    now_script = 0
                else:
                    if now_script < 8:
                        self.list_of_threads.append(ScriptsThread(self.device, [script], neighbours))
                    else:
                        self.list_of_threads[now_thread].scripts.add(script)
                now_thread += 1
                now_script += 1
            
            self.barrier = ReusableBarrierCond(len(self.list_of_threads))
            for thread in self.list_of_threads:
                thread.start()

            for thread in self.list_of_threads:
                thread.join()
            
            self.list_of_threads = []
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
            self.list_of_threads = []
            