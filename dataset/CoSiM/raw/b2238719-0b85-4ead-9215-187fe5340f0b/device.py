


from threading import Event, Thread, Semaphore
from barrier import ReusableBarrierSem
from worker import WorkerThread


class Device(object):
    

    
    def __init__(self, device_id, sensor_data, supervisor):
        

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.barrier_set = Event()
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()


        self.thread = DeviceThread(self)
        
        self.barrier = None
        
        self.neighbours = []
        
        self.data_locks = []
        
        self.thread_list = []
        
        self.worker_number = 8
        
        self.worker_barrier = ReusableBarrierSem(self.worker_number)
        
        self.script_queue = []
        
        self.script_lock = Semaphore(1)
        
        self.exit_flag = Event()
        
        self.tasks_finished = Event()
        
        self.start_tasks = Event()

    def set_flag(self):
        
        self.barrier_set.set()

    def set_barrier(self, barrier):
        
        self.barrier = barrier

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            
            self.barrier = ReusableBarrierSem(len(devices))
            location_index = -1
            for dev in devices:
                for k in dev.sensor_data:
                    if k > location_index:
                        location_index = k

            
            for dev in devices:
                dev.set_barrier(self.barrier)
                dev.set_flag()

            
            self.data_locks = {loc : Semaphore(1) for loc in range(location_index+1)}
            for dev in devices:
                dev.data_locks = self.data_locks
        else:
            
            self.barrier_set.wait()
        self.thread.start()

        
        for tid in range(self.worker_number):
            thread = WorkerThread(self, tid)
            self.thread_list.append(thread)
            thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        
        for thread in self.thread_list:
            thread.join()
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            self.device.barrier.wait()

            
            self.device.neighbours = self.device.supervisor.get_neighbours()

            
            if self.device.neighbours is None:
              
                self.device.exit_flag.set()
                self.device.start_tasks.set()
                break

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            self.device.script_queue = list(self.device.scripts)

            
            self.device.start_tasks.set()

            
            self.device.tasks_finished.wait()
            self.device.tasks_finished.clear()

from threading import Thread
class WorkerThread(Thread):
    
    def __init__(self, device, thread_id):
        Thread.__init__(self)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        iteratii = 0
        while True:

            
            self.device.worker_barrier.wait()

            
            if self.thread_id == 0 and iteratii != 0:
                self.device.tasks_finished.set()

            
            self.device.start_tasks.wait()

            
            self.device.worker_barrier.wait()
            if self.thread_id == 0:
                self.device.start_tasks.clear()
            iteratii += 1
            if self.device.exit_flag.is_set():
                break

            self.device.script_lock.acquire()
            if len(self.device.script_queue) > 0:
                (script, location) = self.device.script_queue.pop(0)
                self.device.script_lock.release()
            else:
                self.device.script_lock.release()
                continue

            self.device.data_locks[location].acquire()
            
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
            self.device.data_locks[location].release()

