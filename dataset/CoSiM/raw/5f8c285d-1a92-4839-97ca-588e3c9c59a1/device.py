


from threading import Thread, Event, Lock, Semaphore


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.lock = Lock()
        self.all_scripts_received = Event()
        self.barrier = None
        self.thread = None
        self.devices = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id is 0:
            self.devices = devices
            
            self.barrier = ReusableBarrier(len(devices))
            self.thread = DeviceThread(self, self.lock, self.barrier)
            for dev in devices:
                if dev.device_id is not 0:


                    dev.barrier = self.barrier
                    
                    dev.thread = DeviceThread(dev, dev.lock, self.barrier)
                
                dev.thread.start()

    def assign_script(self, script, location):
        

        if script is not None:
            
            self.scripts.append((script, location))
        else:
            
            self.all_scripts_received.set()

    def get_data(self, location):
        

        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None


    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        
        if self.device_id is 0:
            for dev in self.devices:
                dev.thread.join()




class DeviceThread(Thread):
    

    def __init__(self, device, lock, barrier):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.lock = lock
        self.barrier = barrier

    def run(self):

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break
            
            self.device.all_scripts_received.wait()
            
            self.device.all_scripts_received.clear()

            
            for (script, location) in self.device.scripts:

                script_data = []
                
                for device in neighbours:
                    
                    device.lock.acquire()
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
                for device in neighbours:
                    
                    device.lock.release()

            
            self.barrier.wait()

class ReusableBarrier():
    
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
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        
        threads_sem.acquire()
 