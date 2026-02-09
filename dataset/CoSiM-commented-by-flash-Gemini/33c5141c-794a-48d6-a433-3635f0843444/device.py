"""
Module: device.py
Description: Semantic documentation for device.py.
             Detailed semantic analysis will be applied later.
"""



from threading import Event, Thread, Lock
from barrier import Barrier
from threadpool import ThreadPool


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

        self.barrier = None
        self.locks = {}
        self.script_received = False

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for location in sensor_data:
            self.locks[location] = Lock()

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
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for device in devices:
                device.barrier = self.barrier

    '''
    Functional Utility: Describe purpose of assign_script here.
    '''
    def assign_script(self, script, location):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if script is not None:
            self.script_received = True
            self.scripts.append((location, script))
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
        if location not in self.sensor_data:
            return None

        self.locks[location].acquire()
        return self.sensor_data[location]

    '''
    Functional Utility: Describe purpose of set_data here.
    '''
    def set_data(self, location, data):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

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
        self.pool = ThreadPool(8)

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
        
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        while True:
            
            local_devices = self.device.supervisor.get_neighbours()
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if local_devices is None:
                break

            
            temp_set = set(local_devices)
            temp_set.add(self.device)
            local_devices = list(temp_set)
            

            
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            while True:

                
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if self.device.script_received or self.device.timepoint_done.wait():


                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    if self.device.script_received:
                        self.device.script_received = False

                        
                        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                        # Invariant: State condition that holds true before and after each iteration/execution
                        for (location, script) in self.device.scripts:
                            self.pool.add_task(location, script, local_devices)
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_received = True
                        break
            
            self.pool.wait_tasks()
            self.device.barrier.wait()

        
        self.pool.join_threads()




from Queue import Queue
from threading import Thread

'''
Functional Utility: Describe purpose of execute_script here.
'''
def execute_script(location, script, local_devices):
    

    data_collection = []

    
    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
    # Invariant: State condition that holds true before and after each iteration/execution
    for device in local_devices:
        data = device.get_data(location)
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if data is not None:
            data_collection.append(data)

    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
    # Invariant: State condition that holds true before and after each iteration/execution
    if data_collection != []:
        result = script.run(data_collection)

        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for device in local_devices:
            device.set_data(location, result)

class ThreadPool(object):

    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, threads_count):
        

        self.tasks = Queue(threads_count)
        self.threads = []

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for _ in xrange(threads_count):
            self.threads.append(Thread(target=self.run))

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for thread in self.threads:
            thread.start()


    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
        

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        while True:

            location, script, local_devices = self.tasks.get()

            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if script is None and local_devices is None:
                self.tasks.task_done()
                return

            execute_script(location, script, local_devices)
            self.tasks.task_done()

    '''
    Functional Utility: Describe purpose of add_task here.
    '''
    def add_task(self, location, script, local_devices):
        

        self.tasks.put((location, script, local_devices))

    '''
    Functional Utility: Describe purpose of wait_tasks here.
    '''
    def wait_tasks(self):
        

        self.tasks.join()

    '''
    Functional Utility: Describe purpose of join_threads here.
    '''
    def join_threads(self):
        

        self.wait_tasks()

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for _ in xrange(len(self.threads)):
            self.add_task(None, None, None)

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for thread in self.threads:
            thread.join()
