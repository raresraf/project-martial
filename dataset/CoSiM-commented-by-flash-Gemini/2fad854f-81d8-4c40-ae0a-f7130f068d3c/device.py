"""
Module: device.py
Description: Semantic documentation for device.py.
             Detailed semantic analysis will be applied later.
"""



from threading import Semaphore, Lock, Event, Thread

class ReusableBarrier():
    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    '''
    Functional Utility: Describe purpose of wait here.
    '''
    def wait(self):
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    '''
    Functional Utility: Describe purpose of phase here.
    '''
    def phase(self, count_threads, threads_sem):
        with self.count_lock:
            count_threads[0] -= 1
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if count_threads[0] == 0:            
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    

class Device(object):
    
    
    
    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()


        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        
        self.hash_ld = {}

    '''
    Functional Utility: Describe purpose of __str__ here.
    '''
    def __str__(self):
        
        return "Device %d" % self.device_id

    
    '''
    Functional Utility: Describe purpose of exists here.
    '''
    def exists(self,key):
        nr = 0;
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for k in self.hash_ld.keys():
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if(k == key):
                nr += 1


        return nr
    '''
    Functional Utility: Describe purpose of setup_devices here.
    '''
    def setup_devices(self, devices):
        
        
        nrd = len(devices)
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if(self.device_id == 0):
            
            self.barrier = ReusableBarrier(nrd)
            
            
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for device in devices:
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if(device.device_id != 0):
                    device.barrier = self.barrier
                    device.hash_ld = self.hash_ld

        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for device in devices:
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for k in device.sensor_data.keys():
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if(self.exists(k) == 0):
                    self.hash_ld[k] = Lock()

    '''
    Functional Utility: Describe purpose of assign_script here.
    '''
    def assign_script(self, script, location):
        
        
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if script is not None:
            self.scripts.append((script, location))
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        else:
            self.timepoint_done.set()

    '''
    Functional Utility: Describe purpose of get_data here.
    '''
    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    '''
    Functional Utility: Describe purpose of set_data here.
    '''
    def set_data(self, location, data):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if location in self.sensor_data:
            self.sensor_data[location] = data

    '''
    Functional Utility: Describe purpose of shutdown here.
    '''
    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.t = []

    
    '''
    Functional Utility: Describe purpose of method here.
    '''
    def method(self,script, script_data):
        result = script.run(script_data)
        return result

    
    '''
    Functional Utility: Describe purpose of method_scripts here.
    '''
    def method_scripts(self,neighbours,script,location):
        
        
         self.device.hash_ld[location].acquire()
         script_data = []
         # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
         # Invariant: State condition that holds true before and after each iteration/execution
         for device in neighbours:
             data = device.get_data(location)
             # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
             # Invariant: State condition that holds true before and after each iteration/execution
             if data is not None:
                 script_data.append(data)
         data = self.device.get_data(location)
         # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
         # Invariant: State condition that holds true before and after each iteration/execution
         if data is not None:
             script_data.append(data)

         # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
         # Invariant: State condition that holds true before and after each iteration/execution
         if script_data != []:
             result = self.method(script,script_data)
             # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
             # Invariant: State condition that holds true before and after each iteration/execution
             for device in neighbours:
                 device.set_data(location, result)
             self.device.set_data(location, result)
         self.device.hash_ld[location].release()

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            self.t = []
            number = len(self.device.scripts)
            number_of_threads = min(8, number)
            nr = 1
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for (script, location) in self.device.scripts:
                
                
                self.t.append(Thread(target = self.method_scripts, args = (neighbours,script,location)))
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if(nr == number_of_threads):
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    for i in range(0,nr):
                        self.t[i].start()
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    for i in range(0,nr):
                        self.t[i].join()
                    self.t = []
                    nr = 0
                nr += 1

            
            
            self.device.barrier.wait()
