"""
Module: device.py
Description: Semantic documentation for device.py.
             Detailed semantic analysis will be applied later.
"""


from threading import Lock, Thread, Semaphore, Event
from Queue import Queue


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
        self.barrier = None

        
        
        self.locks = {}
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for spot in sensor_data:
            self.locks[spot] = Lock()
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
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for dev in devices:
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if dev.device_id != self.device_id:
                    dev.barrier = self.barrier

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
        for loc in self.sensor_data:
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if loc == location:
                self.locks[loc].acquire()
                return self.sensor_data[loc]

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
        
        self.dev_threads = ThreadsForEachDevice(8)

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):

        self.dev_threads.device = self.device

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
                self.dev_threads.jobs_to_be_done.put(
                    (neighbours, script, location))

            self.device.timepoint_done.clear()

            
            self.dev_threads.jobs_to_be_done.join()
            
            self.device.barrier.wait()

        
        self.dev_threads.jobs_to_be_done.join()

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for _ in range(len(self.dev_threads.threads)):
            self.dev_threads.jobs_to_be_done.put((None, None, None))

        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for d_th in self.dev_threads.threads:
            d_th.join()


class ThreadsForEachDevice(object):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, number_of_threads):
        self.device = None
        
        
        
        self.jobs_to_be_done = Queue(number_of_threads)
        self.threads = []

        self.create_threads(number_of_threads)
        self.start_threads()

    '''
    Functional Utility: Describe purpose of create_threads here.
    '''
    def create_threads(self, number_of_threads):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for _ in range(number_of_threads):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

    '''
    Functional Utility: Describe purpose of start_threads here.
    '''
    def start_threads(self):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        for i_th in self.threads:
            i_th.start()

    '''
    Functional Utility: Describe purpose of execute here.
    '''
    def execute(self):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        while True:
            
            neighbours, script, location = self.jobs_to_be_done.get()
            
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if neighbours is None and script is None:
                self.jobs_to_be_done.task_done()
                return

            data_for_script = []
            
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
                        data_for_script.append(data)
            
            data = self.device.get_data(location)
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if data is not None:
                data_for_script.append(data)

            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if data_for_script != []:
                
                scripted_data = script.run(data_for_script)

                
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                for device in neighbours:
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    if device.device_id != self.device.device_id:
                        device.set_data(location, scripted_data)

                
                self.device.set_data(location, scripted_data)

            self.jobs_to_be_done.task_done()
