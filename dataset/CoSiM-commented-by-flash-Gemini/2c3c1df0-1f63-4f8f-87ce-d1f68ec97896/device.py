




from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
from workPool import WorkPool

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.workpool = WorkPool(4, self)

        self.scripts = []
        self.script_storage = []
        self.locks = []
        self.barrier = None
        self.neighbours = None

        self.script_lock = Lock()
        self.supervisor_interact = Event()
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        ids = []
        loc = []

        
        for device in devices:
            ids.append(device.device_id)
            for location, _ in device.sensor_data.iteritems():
                loc.append(location)

        
        max_locations = max(loc) + 1
        if self.device_id == min(ids):
            barrier = ReusableBarrierSem(len(ids))
            locks = [Lock() for _ in range(max_locations)]
            for device in devices:
                device.assign_barrier(barrier)
                device.set_locks(locks)

    def assign_barrier(self, barrier):
        
        self.barrier = barrier

    def assign_script(self, script, location):
        

        if script is not None:
            self.script_lock.acquire()
            self.scripts.append((script, location))
            self.script_lock.release()
        else:
            self.timepoint_done.set()
        self.supervisor_interact.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def wait_on_scripts(self):
        
        self.workpool.wait()

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

    def set_locks(self, locks):
        
        self.locks = locks

    def lock(self, location):
        
        self.locks[location].acquire()

    def unlock(self, location):
        
        self.locks[location].release()

    def execute_scripts(self):
        

        
        self.script_lock.acquire()

        for (script, location) in self.scripts:
            self.script_storage.append((script, location))
            self.workpool.add_data(script, location)

        del self.scripts[:]
        self.script_lock.release()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                self.device.workpool.shutdown()
                return

            
            
            self.device.execute_scripts()

            while True:

                
                self.device.supervisor_interact.wait()
                self.device.supervisor_interact.clear()

                
                self.device.script_lock.acquire()
                if len(self.device.scripts) > 0:
                    self.device.script_lock.release()
                    self.device.execute_scripts()
                else:
                    self.device.script_lock.release()

                
                if self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
                    
                    
                    
                    
                    if len(self.device.scripts) > 0:
                        self.device.execute_scripts()

                    
                    self.device.wait_on_scripts()

                    
                    self.device.barrier.wait()
                    self.device.scripts = self.device.script_storage
                    self.device.script_storage = []
                    break

from threading import Thread




class ScriptExecutor(Thread):
    

    def __init__(self, index, workpool, device):
        

        Thread.__init__(self, name="Worker Thread %d" % index)
        self.index = index
        self.workpool = workpool
        self.device = device

    def run(self):
        while True:
            self.workpool.data.acquire()
            if self.workpool.done:
                return

            (script, location) = self.workpool.q.get()

            if self.device.neighbours is None:
                
                return

            self.device.lock(location)

            script_data = []
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                for device in self.device.neighbours:
                    device.set_data(location, result)

                
                self.device.set_data(location, result)

            self.device.unlock(location)
            self.workpool.q.task_done()

from threading import Semaphore
from Queue import Queue
from scriptexecutor import ScriptExecutor

class WorkPool(object):
    



    def __init__(self, num_threads, device):
        
        self.device = device
        self.executors = []
        self.q = Queue()
        self.data = Semaphore(0)
        self.done = False

        for i in range(num_threads + 1):
            executor = ScriptExecutor(i, self, self.device)
            executor.start()
            self.executors.append(executor)

    def add_data(self, script, location):
        

        self.q.put((script, location))
        self.data.release()

    def wait(self):
        
        if not self.done:
            self.q.join()

    def shutdown(self):
        
        self.wait()
        self.done = True

        for _ in self.executors:
            self.data.release()

        for executor in self.executors:
            executor.join()
