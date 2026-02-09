"""
Module: device.py
Description: Semantic documentation for device.py.
             Detailed semantic analysis will be applied later.
"""



from threading import Event, Lock
from utils import ReusableBarrierSem
from device_thread import DeviceThread

class Device(object):
    

    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.locks = {}
        self.barrier = None
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


            self.barrier = ReusableBarrierSem(len(devices))
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for current_device in devices:
                current_device.barrier = self.barrier
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                for location in current_device.sensor_data:
                    # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                    # Invariant: State condition that holds true before and after each iteration/execution
                    if self.locks.has_key(location) is False:


                        self.locks[location] = Lock()
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for current_device in devices:
                current_device.locks = self.locks

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

from threading import Thread
from multiprocessing.dummy import Pool

class DeviceThread(Thread):
    
    '''
    Functional Utility: Describe purpose of __init__ here.
    '''
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        
        self.num_threads = 8
        self.device = device
        
        self.threads_pool = Pool(self.num_threads)
        self.neighbours = None

    '''
    Functional Utility: Describe purpose of calculate here.
    '''
    def calculate(self, raw_data):
        
        data_list = []
        with self.device.locks[raw_data[1]]:
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            for i in range(len(self.neighbours)):
                current_data = self.neighbours[i].get_data(raw_data[1])
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                if current_data is None:
                    continue
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                else:


                    data_list.append(current_data)

            my_data = self.device.get_data(raw_data[1])
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if my_data is not None:
                data_list.append(my_data)

            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if data_list != []:
                new_data = raw_data[0].run(data_list)
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                for i in range(len(self.neighbours)):
                    self.neighbours[i].set_data(raw_data[1], new_data)
                self.device.set_data(raw_data[1], new_data)

    '''
    Functional Utility: Describe purpose of run here.
    '''
    def run(self):
        
        # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
        # Invariant: State condition that holds true before and after each iteration/execution
        while True:
            self.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            if self.neighbours is not None:
                self.device.timepoint_done.wait()
                self.threads_pool.map(self.calculate, self.device.scripts)
                self.device.barrier.wait()
                self.device.timepoint_done.clear()
            # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
            # Invariant: State condition that holds true before and after each iteration/execution
            else:
                break
        
        self.threads_pool.close()
        self.threads_pool.join()

from threading import Lock, Semaphore

class ReusableBarrierSem(object):
    
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
                i = 0
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                while i < self.num_threads:
                    i += 1
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
                i = 0
                # Block Logic: Describe purpose of this block, e.g., iteration, conditional execution
                # Invariant: State condition that holds true before and after each iteration/execution
                while i < self.num_threads:
                    i += 1
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()
