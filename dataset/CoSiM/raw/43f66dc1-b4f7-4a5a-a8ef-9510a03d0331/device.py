

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


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
        self.barrier = None
        self.big_lock = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        barrier = ReusableBarrier(len(devices))
        lock1 = Lock()

        num_locations = {}

        for device in devices:
            for location in device.sensor_data.keys():
                num_locations[location] = 1

        big_lock = [Lock() for _ in range(len(num_locations))]

        for device in devices:
            device.lock1 = lock1
            device.barrier = barrier
            device.big_lock = big_lock

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] \
            if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class ParallelScript(Thread):
    
    def __init__(self, device, scripts, location, neighbours):
        
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts
        self.location = location
        self.neighbours = neighbours

    def run(self):

        for script in self.scripts:
            self.device.big_lock[self.location].acquire()


            
            
            

            script_data = []
            
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)

            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                
                for device in self.neighbours:


                    device.set_data(self.location, result)

                
                self.device.set_data(self.location, result)
            self.device.big_lock[self.location].release()


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
            scripts = {}

            
            for (script, location) in self.device.scripts:
                if scripts.has_key(location):
                    scripts[location].append(script)
                else:
                    scripts[location] = [script]

            
            for location in scripts.keys():
                new = ParallelScript(self.device, scripts[location],
                                     location, neighbours)


                threads.append(new)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            
            self.device.barrier.wait()

            self.device.timepoint_done.clear()
