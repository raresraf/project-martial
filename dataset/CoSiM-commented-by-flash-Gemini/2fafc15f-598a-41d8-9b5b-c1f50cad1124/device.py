"""
Module: device.py
Description: Semantic documentation for device.py.
             Detailed semantic analysis will be applied later.
"""



from threading import Event, Thread, Condition, Lock

class ReusableBarrier(object):
    
    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
    '''
    Functional Utility: Describe purpose of wait here.
    '''
    def wait(self):
        
        self.cond.acquire()
        self.count_threads -= 1
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        else:
            self.cond.wait()
        self.cond.release()

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
        self.barr = None
        self.lock = None
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
        
        


        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if devices[0].barr is None and devices[0].device_id == self.device_id:
            bariera = ReusableBarrier(len(devices))
            lock = Lock()
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for i in devices:
                i.barr = bariera
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for j in devices:
                j.lock = lock
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
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if location in self.sensor_data:
            return self.sensor_data[location]
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        else:
            return None

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

            
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for (script, location) in self.device.scripts:
                self.device.lock.acquire()
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
                self.device.lock.release()

            
            self.device.timepoint_done.clear()
            self.device.barr.wait()
