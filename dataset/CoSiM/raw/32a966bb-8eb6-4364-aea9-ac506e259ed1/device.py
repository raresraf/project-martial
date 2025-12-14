


from threading import Event, Thread, Condition


class ReusableBarrier():
    
    num_threads = 0
    count_threads = 0

    def __init__(self):
        
        self.cond = Condition()
        self.thread_event = Event()

    def wait(self):
        
        self.cond.acquire()
        ReusableBarrier.count_threads -= 1

        if ReusableBarrier.count_threads == 0:
            self.cond.notify_all()
            ReusableBarrier.count_threads = ReusableBarrier.num_threads
        else:
            self.cond.wait()

        self.cond.release()

    @staticmethod
    def add_thread():
        
        ReusableBarrier.num_threads += 1
        ReusableBarrier.count_threads = ReusableBarrier.num_threads


class Device(object):
    
    barr = ReusableBarrier()    

    def __init__(self, device_id, sensor_data, supervisor):
        
        Device.barr.add_thread()

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        pass

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

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
            
            Device.barr.wait()

            self.device.script_received.wait()
            self.device.script_received.clear()

            
            for (script, location) in self.device.scripts:
                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    for device in neighbours:
                        device.set_data(location, result)

                    self.device.set_data(location, result)
