


from threading import Event, Thread, Condition, Lock


class Barrier(object):
    

    def __init__(self, num_threads=0):
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
    
    
    bariera_devices = Barrier()
    locks = []

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        
        


        self.scripts = []
        self.locations = []
        
        self.nr_scripturi = 0
        
        self.script_crt = 0

        
        
        self.timepoint_done = Event()

        
        self.neighbours = []
        self.event_neighbours = Event()
        self.lock_script = Lock()
        self.bar_thr = Barrier(8)

        
        self.thread = DeviceThread(self, 1)
        self.thread.start()
        self.threads = []
        for _ in range(7):
            tthread = DeviceThread(self, 0)
            self.threads.append(tthread)
            tthread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        Device.bariera_devices = Barrier(len(devices))
        
        if Device.locks == []:
            for _ in range(self.supervisor.supervisor.testcase.num_locations):
                Device.locks.append(Lock())

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append(script)
            self.locations.append(location)
            
            self.nr_scripturi += 1
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()
        for tthread in self.threads:
            tthread.join()


class DeviceThread(Thread):
    

    def __init__(self, device, first):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.first = first

    def run(self):
        while True:
            
            
            if self.first == 1:
                
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.script_crt = 0


                self.device.event_neighbours.set()

            
            self.device.event_neighbours.wait()

            if self.device.neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            while True:
                
                self.device.lock_script.acquire()
                index = self.device.script_crt
                self.device.script_crt += 1
                self.device.lock_script.release()



                
                
                if index >= self.device.nr_scripturi:
                    break

                
                location = self.device.locations[index]
                script = self.device.scripts[index]

                
                
                Device.locks[location].acquire()

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

                Device.locks[location].release()

            
            self.device.bar_thr.wait()
            
            if self.first == 1:
                self.device.event_neighbours.clear()
                self.device.timepoint_done.clear()
            self.device.bar_thr.wait()
            
            if self.first == 1:
                Device.bariera_devices.wait()

