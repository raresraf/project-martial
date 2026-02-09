"""
Module: device.py
Description: Semantic documentation for device.py.
             Detailed semantic analysis will be applied later.
"""



from threading import Event, Thread, Semaphore, Lock

class Device(object):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_scripts = []
        self.neighbours = []
        self.timepoint_done = Event()
        
        self.initialization = Event()
        
        self.threads = []
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for k in xrange(8):
            self.threads.append(DeviceThread(self, k))
        self.locations_lock = Lock()
        self.locked_locations = None
        self.devices_barrier = None
        self.device_barrier = ReusableBarrier(len(self.threads))

    '''
    Functional Utility: Describe purpose of __str__ here.
    '''
    def __str__(self):
        
        return "Device %d" % self.device_id

    '''
    Functional Utility: Describe purpose of setup_devices here.
    '''
    def setup_devices(self, devices):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if self.device_id == 0:
            
            self.locked_locations = {}

            
            self.devices_barrier = ReusableBarrier(len(devices)*len(self.threads))

            
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for device in devices:
                device.locked_locations = self.locked_locations
                device.devices_barrier = self.devices_barrier


                device.initialization.set()

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        else:
            
            self.initialization.wait()

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for thread in self.threads:
            thread.start()

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
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        while True:
            
            self.device.devices_barrier.wait()

            
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            


            self.device.device_barrier.wait()
            neighbours = self.device.neighbours
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if self.thread_id == 0:
                self.device.timepoint_scripts = self.device.scripts[:]
            self.device.device_barrier.wait()
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            while True:
                
                self.device.locations_lock.acquire()
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if len(self.device.timepoint_scripts) == 0:
                    self.device.locations_lock.release()
                    break
                (script, location) = self.device.timepoint_scripts.pop()

                
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if location not in self.device.locked_locations:


                    self.device.locked_locations[location] = Lock()

                self.device.locked_locations[location].acquire()
                self.device.locations_lock.release()

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
                    
                    result = script.run(script_data)

                    
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                
                self.device.locked_locations[location].release()

            
            self.device.device_barrier.wait()
            self.device.timepoint_done.clear()

class ReusableBarrier(object):
    
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
                for _ in range(self.num_threads):
                    
                    threads_sem.release()
                count_threads[0] = self.num_threads        
        
        threads_sem.acquire()                    
