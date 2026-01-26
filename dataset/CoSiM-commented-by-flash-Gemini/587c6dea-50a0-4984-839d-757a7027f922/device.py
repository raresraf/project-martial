


from threading import Event, Lock, Thread, Condition


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
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.master = None
        self.bariera = None
        self.lock = Lock()
        self.lacate = {}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.master = devices[0]
        self.master.bariera = ReusableBarrierCond(len(devices))


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
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

            self.device.script_received.wait()

            tlist = []
            
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

                    my_thread = MyThread(script, script_data, location,
                    neighbours, self.device, self.device.master.lock,
                    self.device.master.lacate)
                    my_thread.start()
                    tlist.append(my_thread)

            for my_thread in tlist:
                my_thread.join()

            self.device.master.bariera.wait()
            self.device.timepoint_done.wait()
            self.device.script_received.clear()

class MyThread(Thread):
    

    def __init__(self, script, script_data, location, neighbours,
                own_device, mlock, lacate):
        
        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        self.location = location
        self.neighbours = neighbours
        self.own_device = own_device
        self.mlock = mlock
        self.lacate = lacate

    def run(self):
        result = self.script.run(self.script_data)

        if self.location in self.lacate:
            self.lacate[self.location].acquire()
        else:
            self.lacate[self.location] = Lock()


            self.lacate[self.location].acquire()

        
        for device in self.neighbours:
            device.set_data(self.location, result)
        
        self.own_device.set_data(self.location, result)

        self.lacate[self.location].release()
