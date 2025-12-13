


from threading import Event, Thread, Lock
from Queue import Queue
import reusable_barrier_semaphore

class Device(object):
    
    
    barrier = None
    
    
    
    lockList = {}
    
    lockListLock = Lock()
    def __init__(self, device_id, sensor_data, supervisor):
        

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        
        self.neighbours_done = reusable_barrier_semaphore.ReusableBarrier(8)
        self.neighbours = None
        
        self.scripts = Queue()
        
        
        
        
        self.permanent_scripts = []
        self.threads = []
        self.startup_event = Event()
        for i in range(0, 8):
            self.threads.append(DeviceThread(self, i))
        for i in range(0, 8):
            self.threads[i].start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        
        
        Device.lockListLock.acquire()
        if Device.barrier is None:
            Device.barrier = reusable_barrier_semaphore.ReusableBarrier(len(devices))
        Device.lockListLock.release()

        the_keys = self.sensor_data.keys()

        for i in the_keys:
            Device.lockListLock.acquire()
            
            
            
            if i not in Device.lockList:
                Device.lockList[i] = Lock()
            Device.lockListLock.release()
        
        
        self.startup_event.set()

    def assign_script(self, script, location):
        
        
        
        
        if script is not None:
            self.scripts.put((script, location))


            self.permanent_scripts.append((script, location))
        else:
            for i in range(0, 8):
                
                
                
                self.scripts.put((script, location))

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for i in range(0, 8):
            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, the_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        
        
        
        self.the_id = the_id



    def run(self):
        
        
        
        
        self.device.startup_event.wait()
        while True:
            
            
            if self.the_id == 0:
                
                Device.barrier.wait()
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            
            self.device.neighbours_done.wait()

            if self.device.neighbours is None:
                
                break

            
            while True:
                (script, location) = self.device.scripts.get()
                script_data = []
                if script is None:
                    
                    
                    self.device.neighbours_done.wait()
                    if self.the_id == 0:
                        
                        
                        
                        
                        for (script, location) in self.device.permanent_scripts:
                            self.device.scripts.put((script, location))
                    break

                
                
                if location is not None:
                    Device.lockList[location].acquire()
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
                    Device.lockList[location].release()

from threading import Semaphore, Lock
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
