


from threading import Event, Thread

from threading import Condition, RLock    




class ReusableBarrier():
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
 
    def wait(self):
        self.cond.acquire()                      
        self.count_threads -= 1;
        if self.count_threads == 0:
            self.cond.notify_all()              
            self.count_threads = self.num_threads    
        else:
            self.cond.wait();                    
        self.cond.release();                   




class DeviceThread_Worker(Thread):
    def __init__(self, device, neighbours, tid, scripts):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.device = device
        self.scripts = scripts
        self.tid = tid 

    def run(self):
        
        for (script, location) in self.scripts:

            
            script_data = []
            index = location

            
            for device in self.neighbours:

                self.device.locks[index].acquire()
                self.device.lock.acquire()

                data = device.get_data(location)

                self.device.lock.release()
                self.device.locks[index].release()

                if data is not None:
                    script_data.append(data)

            
            
            self.device.locks[index].acquire()
            self.device.lock.acquire()

            data = self.device.get_data(location)

            self.device.lock.release()
            self.device.locks[index].release()

            if data is not None:
                script_data.append(data)

            
            if script_data != []:
                result = script.run(script_data)

                self.device.locks[index].acquire()

                
                for dev in self.neighbours:
                    if result > dev.get_data(location):
                        dev.set_data(location, result)
                    
                
                self.device.set_data(location, result)



                self.device.locks[index].release()   


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.lock = RLock()
        self.barrier = None
        self.devices = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.locations = []
        self.locks = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        self.devices = devices
        if self.device_id == 0:
            for num in range(0, 1000):
                lock = RLock()
                for i in range (0, len(devices)):
                    devices[i].locks.append(lock)
            
            barrier = ReusableBarrier(len(devices)) 
            for i in range(0,len(devices)):
                if devices[i].barrier == None:
                    devices[i].barrier = barrier


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location)) 
            self.timepoint_done.set()
            
        else:
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()
    

    def shutdown(self):
        
        self.thread.join()


 


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def divide_in_threads(self, neighbours):

        

        
        threads = []

        
        nr = len(self.device.scripts)
        numar = 1 
        if nr > 8:
            numar = nr / 8
            nr = 8

        
        for i in range(0,nr):
            if i == nr - 1:
                t = DeviceThread_Worker(self.device, neighbours, i, self.device.scripts[i * numar : len(self.device.scripts)])
            else:
                t = DeviceThread_Worker(self.device, neighbours, i, self.device.scripts[i * numar : i*numar + numar])
            threads.append(t)

        
        for i in range(0, nr):
            threads[i].start()

        for i in range(0,nr):


            threads[i].join()

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.script_received.wait()

            
            
            self.divide_in_threads(neighbours)


            
            self.device.script_received.clear()


            
            self.device.barrier.wait()

 
