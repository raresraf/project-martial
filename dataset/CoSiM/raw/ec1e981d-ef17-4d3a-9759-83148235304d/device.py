




from _threading_local import local
from threading import Event, Thread, Lock, RLock, Condition, Semaphore
from barrier import *

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.lock = Lock()
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

        self.sync_barrier = None
        self.devices = []
        self.location_locks = {}

        self.nbs = []


    def __str__(self):
        
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        
        self.sync_barrier = barrier

    def sync_with_others(self):
        
        self.sync_barrier.wait()

    def set_locks(self, locks):
        
        self.location_locks = locks

    def get_lock(self, location):
        
        return self.location_locks[location]

    def setup_devices(self, devices):
        
        self.devices = devices
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            locks = {}
            for dev in devices:
                for loc in dev.sensor_data:
                    locks[loc] = Lock()
                dev.set_barrier(barrier)
            for dev in devices:
                dev.set_locks(locks)

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_scripts(self):
        
        return [(self, s, l, self.nbs) for (s, l) in self.scripts]

    def get_data(self, l):
        
        self.lock.acquire()
        ret = self.sensor_data[l] if l in self.sensor_data else None
        self.lock.release()
        return ret

    def set_data(self, location, data):
        
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()
    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    
    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.max_running_threads_cnt = 8

    @staticmethod
    def exec_script(invoker_device, script, location, neighbourhood):
        
        script_data = []
        invoker_device.location_locks[location].acquire()
        
        for device in neighbourhood:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = invoker_device.get_data(location)
        if data is not None:
            script_data.append(data)
        if script_data != []:
            
            result = script.run(script_data)
            
            for device in neighbourhood:
                device.set_data(location, result)
            
            invoker_device.set_data(location, result)
        invoker_device.location_locks[location].release()

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.nbs = neighbours
            
            self.device.timepoint_done.wait()

            
            self.device.sync_with_others()
            
            scrpts = []
            threads = []

            
            scrpts.extend(self.device.get_scripts())
            running_threads_cnt = 0

            
            for (d, s, l, n) in scrpts:
                thread = Thread(name="T",
                                target=DeviceThread.exec_script,
                                args=(d, s, l, n))
                threads.append(thread)
                thread.start()
                running_threads_cnt += 1
                
                
                if running_threads_cnt >= self.max_running_threads_cnt:
                    wthread = threads.pop(0)
                    running_threads_cnt -= 1
                    wthread.join()

            
            for thread in threads:
                thread.join()

            
            self.device.timepoint_done.clear()
            
            
