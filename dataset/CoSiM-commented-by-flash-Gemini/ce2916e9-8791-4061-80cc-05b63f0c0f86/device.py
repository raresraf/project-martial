


from threading import Event, Semaphore, Lock, Thread
from Queue import Queue

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.no_th = 8

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        

        if self.device_id == 0:
            barrier = ReusableBarrierSem(len(devices))
            lock_for_loct = {}
            for device in devices:
                device.barrier = barrier
                for location in device.sensor_data:
                    if location not in lock_for_loct:
                        lock_for_loct[location] = Lock()
                device.lock_for_loct = lock_for_loct

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set()

    def get_data(self, location):
        



        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data, source=None):
        

        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = []
        self.neighbours = []

    def run(self):
        
        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break

            self.device.scripts_received.wait()
            self.device.scripts_received.clear()
            
            self.queue = Queue()
            for script in self.device.scripts:
                self.queue.put_nowait(script)

            
            for _ in range(self.device.no_th):
                SolveScript(self.device, self.neighbours, self.queue).start()
            


            self.queue.join()
            
            self.device.barrier.wait()

class SolveScript(Thread):
    

    def __init__(self, device, neighbours, queue):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.queue = queue

    def run(self):
        
        try:
            for (script, location) in self.device.scripts:
                (script, location) = self.queue.get(False)
                
                self.device.lock_for_loct[location].acquire()

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

                
                self.device.lock_for_loct[location].release()
                
                self.queue.task_done()
        except:
            pass

class ReusableBarrierSem():
    

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        
        self.phase1()
        self.phase2()

    def phase1(self):
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()
