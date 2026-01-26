


from threading import Event, Lock
from utils import ReusableBarrierSem
from device_thread import DeviceThread

class Device(object):
    

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

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:


            self.barrier = ReusableBarrierSem(len(devices))
            for current_device in devices:
                current_device.barrier = self.barrier
                for location in current_device.sensor_data:
                    if self.locks.has_key(location) is False:


                        self.locks[location] = Lock()
            for current_device in devices:
                current_device.locks = self.locks

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

from threading import Thread
from multiprocessing.dummy import Pool

class DeviceThread(Thread):
    
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        
        self.num_threads = 8
        self.device = device
        
        self.threads_pool = Pool(self.num_threads)
        self.neighbours = None

    def calculate(self, raw_data):
        
        data_list = []
        with self.device.locks[raw_data[1]]:
            for i in range(len(self.neighbours)):
                current_data = self.neighbours[i].get_data(raw_data[1])
                if current_data is None:
                    continue
                else:


                    data_list.append(current_data)

            my_data = self.device.get_data(raw_data[1])
            if my_data is not None:
                data_list.append(my_data)

            if data_list != []:
                new_data = raw_data[0].run(data_list)
                for i in range(len(self.neighbours)):
                    self.neighbours[i].set_data(raw_data[1], new_data)
                self.device.set_data(raw_data[1], new_data)

    def run(self):
        
        while True:
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is not None:
                self.device.timepoint_done.wait()
                self.threads_pool.map(self.calculate, self.device.scripts)
                self.device.barrier.wait()
                self.device.timepoint_done.clear()
            else:
                break
        
        self.threads_pool.close()
        self.threads_pool.join()

from threading import Lock, Semaphore

class ReusableBarrierSem(object):
    
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
                i = 0
                while i < self.num_threads:
                    i += 1
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                i = 0
                while i < self.num_threads:
                    i += 1
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()
