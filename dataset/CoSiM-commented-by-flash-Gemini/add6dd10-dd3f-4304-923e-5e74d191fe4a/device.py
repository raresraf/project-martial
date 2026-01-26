


from threading import Event, Thread, Lock, Condition


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.data_locks = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:

            new_bar = Barrier(len(devices))

            locations = []
            for device in devices:
                for (location, value) in device.sensor_data.items():
                    locations.append(location)

            max_loc = max(locations)
            data_locks = []
            for i in range(max_loc + 1):
                data_locks.append(Lock())
            
            for device in devices:
                device.set_barrier_locks(new_bar, data_locks)

    def set_barrier_locks(self, barrier, data_locks):
        
        self.barrier = barrier
        self.data_locks = data_locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
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

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            threads = []
            scripts = []

            for i in range(8):
                scripts.append([])

            count = 0
            
            for (script, location) in self.device.scripts:
                scripts[count % 8].append((script, location))
                count += 1

            
            for i in range(8):
                if len(scripts[i]) > 0:


                    thread = ScriptThread(self.device, scripts[i], neighbours)
                    thread.start()
                    threads.append(thread)

            
            for i in range(len(threads)):
                threads[i].join()

            
            self.device.barrier.wait()
            self.device.timepoint_done.clear()


class ScriptThread(Thread):
    

    def __init__(self, device, scripts, neighbours):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        
        
        for (script, location) in self.scripts:



            self.device.data_locks[location].acquire()

            script_data = []

            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
            if script_data != []:

                result = script.run(script_data)



                for device in self.neighbours:
                    device.set_data(location, result)

                self.device.set_data(location, result)

            self.device.data_locks[location].release()


class Barrier():
    

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
