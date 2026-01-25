


from threading import Event, Thread, Lock, Semaphore
from multiprocessing import cpu_count


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
                                                 


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

    def __str__(self):
        
        return "Device %d" % self.device_id

    def set_lock(self, lock1, lock2, barrier1, barrier2):
        self.lock1=lock1
        self.lock2=lock2    
        self.script_received = barrier1
        self.timepoint_done = barrier2


    def setup_devices(self, devices):
        

        
        if self.device_id==devices[0].device_id:
            lock1=Lock()
            lock2=Lock()
            barrier1=ReusableBarrier(len(devices))
            barrier2=ReusableBarrier(len(devices))
            for dev in devices:
                dev.set_lock(lock1, lock2, barrier1, barrier2)

        self.thread = DeviceThread(self)
        self.thread.start()        

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class MyThread(Thread):
    def __init__(self, script, script_date, device, neighbours, location):
        Thread.__init__(self)
        self.script=script
        self.script_data=script_date
        self.result=None
        self.device=device
        self.neighbours=neighbours
        self.location=location

    def run(self):
        result = self.script.run(self.script_data)
        


        self.device.lock2.acquire()
        for device in self.neighbours:
            device.set_data(self.location, result)
        
        self.device.set_data(self.location, result)
        self.device.lock2.release()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        while True:
            
            self.device.lock1.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.lock1.release()


            if neighbours is None:
                break
    
            self.device.script_received.wait()
            
            threads=[]
            
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
                        
                    threads.append(MyThread(script,script_data,self.device,neighbours,location))

            step=cpu_count()*2
            for i in range(0,len(threads),step):
                for j in range(step):
                    if i+j<len(threads):
                        threads[i+j].start()
                for j in range(step):
                    if i+j<len(threads):
                        threads[i+j].join()

            
            self.device.timepoint_done.wait()
