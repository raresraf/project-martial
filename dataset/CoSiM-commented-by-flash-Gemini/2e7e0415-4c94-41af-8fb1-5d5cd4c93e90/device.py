"""
Module: device.py
Description: Semantic documentation for device.py.
             Detailed semantic analysis will be applied later.
"""



from threading import Event, Thread, Lock
from utility import ReusableBarrierCond, ThreadCollection

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

        
        self.barrier = None

        
        self.locks = {}

        
        self.setup = Event()

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
        for location in self.sensor_data:
            self.locks[location] = Lock()

        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))

            
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for device in devices:
                device.barrier = barrier

            
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for device in devices:
                device.setup.set()

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
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]

        return None

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

        
        self.worker_threads = ThreadCollection(self.device, 8)

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
        
        self.device.setup.wait()

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
                self.worker_threads.add_task(script, location, neighbours)

            
            self.device.timepoint_done.clear()

            
            self.worker_threads.queue.join()

            
            self.device.barrier.wait()

        
        self.worker_threads.end_workers()


from Queue import Queue
from threading import Condition, Thread

class ReusableBarrierCond(object):
    

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


class ThreadCollection(object):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, device, num_threads):
        
        self.device = device
        self.threads = []

        
        self.queue = Queue(num_threads)

        
        self.create_workers(num_threads)

        
        self.start_workers()

    '''
    Functional Utility: Describe purpose of __str__ here.
    '''
    def __str__(self):
        
        return "Thread collection belonging to device %d" % self.device.device_id

    '''
    Functional Utility: Describe purpose of create_workers here.
    '''
    def create_workers(self, num_threads):
        
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for _ in xrange(num_threads):
            new_thread = Thread(target=self.run_tasks)
            self.threads.append(new_thread)

    '''
    Functional Utility: Describe purpose of start_workers here.
    '''
    def start_workers(self):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for thread in self.threads:
            thread.start()

    '''
    Functional Utility: Describe purpose of run_tasks here.
    '''
    def run_tasks(self):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        while True:
            
            (neighbours, script, location) = self.queue.get()

            
            
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if location is None and neighbours is None and script is None:
                self.queue.task_done()
                break

            


            self.run_script(script, location, neighbours)
            self.queue.task_done()

    '''
    Functional Utility: Describe purpose of run_script here.
    '''
    def run_script(self, script, location, neighbours):
        
        
        script_data = []

        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for device in neighbours:
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if device.device_id != self.device.device_id:
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
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

            
            self.device.set_data(location, result)

    '''
    Functional Utility: Describe purpose of add_task here.
    '''
    def add_task(self, script, location, neighbours):
        


        self.queue.put((neighbours, script, location))

    '''
    Functional Utility: Describe purpose of end_workers here.
    '''
    def end_workers(self):
        
        
        self.queue.join()

        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for _ in xrange(len(self.threads)):
            self.add_task(None, None, None)

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for thread in self.threads:
            thread.join()
