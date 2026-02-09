"""
Module: device.py
Description: Semantic documentation for device.py.
             Detailed semantic analysis will be applied later.
"""



from threading import Event, Thread, Lock, Semaphore


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
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    '''
    Functional Utility: Describe purpose of __str__ here.
    '''
    def __str__(self):
        
        return "Device %d" % self.device_id

    '''
    Functional Utility: Describe purpose of setup_devices here.
    '''
    def setup_devices(self, devices):
        
        
        flag = True
        device_number = len(devices)

        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for dev in devices:
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if self.device_id > dev.device_id:
                flag = False

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if flag == True:
            barrier = ReusableBarrierSem(device_number)
            map_locations = {}
            tmp = {}
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for dev in devices:
                dev.barrier = barrier
                tmp = list(set(dev.sensor_data) - set(map_locations))
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                for i in tmp:
                    map_locations[i] = Lock()
                dev.map_locations = map_locations
                tmp = {}

    '''
    Functional Utility: Describe purpose of assign_script here.
    '''
    def assign_script(self, script, location):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
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
        
        Thread.__init__(self)
        self.device = device

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        while True:
            
            self.device.timepoint_done.clear()
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if neighbours is None:
                break
            self.device.timepoint_done.wait()
            script_list = []
            thread_list = []
            index = 0
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for script in self.device.scripts:
                script_list.append(script)
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for i in xrange(8):
                thread = SingleDeviceThread(self.device, script_list, neighbours, index)
                thread.start()
                thread_list.append(thread)
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for i in xrange(len(thread_list)):
                thread_list[i].join()
            self.device.barrier.wait()

class SingleDeviceThread(Thread):
    
    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, device, script_list, neighbours, index):
        Thread.__init__(self)
        self.device = device
        self.script_list = script_list
        self.neighbours = neighbours
        self.index = index

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
      
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if self.script_list != []:
            (script, location) = self.script_list.pop(self.index)
            self.compute(script, location)

    '''
    Functional Utility: Describe purpose of update here.
    '''
    def update(self, result, location):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for device in self.neighbours:
            device.set_data(location, result)
        self.device.set_data(location, result)

    '''
    Functional Utility: Describe purpose of collect here.
    '''
    def collect(self, location, neighbours, script_data):
        
        self.device.map_locations[location].acquire()
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for device in self.neighbours:
            
            data = device.get_data(location)
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if data is None:
                pass
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            else:
                script_data.append(data)

        
        data = self.device.get_data(location)
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if data is not None:
            script_data.append(data)

    '''
    Functional Utility: Describe purpose of compute here.
    '''
    def compute(self, script, location):
        
        script_data = []
        self.collect(location, self.neighbours, script_data)

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if script_data == []:
            pass
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        else:
            
            result = script.run(script_data)
            self.update(result, location)

        self.device.map_locations[location].release()

class ReusableBarrierSem():
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    '''
    Functional Utility: Describe purpose of wait here.
    '''
    def wait(self):
        
        self.phase1()
        self.phase2()

    '''
    Functional Utility: Describe purpose of phase1 here.
    '''
    def phase1(self):
        
        with self.counter_lock:
            self.count_threads1 -= 1
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if self.count_threads1 == 0:
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                for i in range(self.num_threads):


                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    '''
    Functional Utility: Describe purpose of phase2 here.
    '''
    def phase2(self):
        
        with self.counter_lock:
            self.count_threads2 -= 1
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if self.count_threads2 == 0:
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()