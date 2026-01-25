


from threading import Thread, Event, Semaphore, Lock

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


class Singleton(object):
    

    class RealSingleton(object):
        

        barrier = None
        locks = None

        def initialize(self, devices):
            

            self.barrier = ReusableBarrier(devices)
            self.locks = {}

        def get_lock(self, location):
            
            if location not in self.locks:
                self.locks[location] = Lock()

            return self.locks[location]

    
    __instance = None

    def __init__(self, numberOfDevices):
        

        if Singleton.__instance is None:
            Singleton.__instance = Singleton.RealSingleton()
            Singleton.__instance.initialize(numberOfDevices)

    def __getattr__(self, attr):
        
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        
        return setattr(self.__instance, attr, value)

    def get_instance(self):
        
        return self.__instance

    def get_lock(self, location):
        
        return self.__instance.get_lock(location)

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):

        self.singleton = None
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        
        
        
        
        self.singleton = Singleton(len(devices))

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return (self.sensor_data[location] if location in self.sensor_data
                else None)

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run_script(self, location, neighbours, script):
        
        script_data = []
        with self.device.singleton.get_lock(location):
            
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


    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            
            
            
            
            self.device.timepoint_done.wait()

            
            
            threads = [Thread(target=self.run_script, args=(
                l, neighbours, s)) for (s, l) in self.device.scripts]

            


            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            
            
            
            
            
            self.device.timepoint_done.clear()
            self.device.singleton.barrier.wait()
