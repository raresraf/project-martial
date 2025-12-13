


from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrier
from Queue import Queue


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = Queue()
        self.lock_device_method = Lock();
        self.neighbours = None
        self.local_barrier = ReusableBarrier(8)
        self.synch_sem = Semaphore(0)
        self.devices = []
        self.startup_event = Event()

        self.vector_of_threads = []
        for i in range(0, 8):
            dev_thread = DeviceThread(self, i)
            self.vector_of_threads.append(dev_thread)
            dev_thread.start();

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        self.devices = devices
        self.startup_event.set()

    def assign_script(self, script, location):
        
        
        if script is not None:


            self.scripts.put((script, location))
        else:
            for _ in range(0, 8):
                self.scripts.put((None, None))


    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for dev_thread in self.vector_of_threads:
            dev_thread.join()

    def synchronize_devices(self):
        
        for device in self.devices:
            device.synch_sem.release()
        for _ in self.devices:
            self.synch_sem.acquire()


class DeviceThread(Thread):
    

    def __init__(self, device, thread_id):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        

        self.device.startup_event.wait()
        
        while True:
            

            if self.thread_id == 0:
                self.device.synchronize_devices()
                self.device.neighbours = self.device.supervisor.get_neighbours()

            self.device.local_barrier.wait()
            if self.device.neighbours is None:
                break

            
            
            
            while True: 
                (script, location) = self.device.scripts.get()
                script_data = []
                if script is None:
                    break
                
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

            
            self.device.local_barrier.wait()
            