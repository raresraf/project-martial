


from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
from taskscheduler import TaskScheduler
from task import Task


def add_lock_for_location(lock_per_location, location):
    

    lock_per_location.append((location, Lock()))

def get_lock_for_location(lock_per_location, location):
    

    for (loc, lock) in lock_per_location:
        if loc == location:
            return lock
    return None


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None
        self.taskscheduler = None
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        

        return "Device %d" % self.device_id

    def share_barrier(self, barrier):
        

        self.barrier = barrier

    def share_taskscheduler(self, taskscheduler):
        

        self.taskscheduler = taskscheduler

    def setup_devices(self, devices):
        

        
        if self.device_id == 0:

            lock_per_location = []

            
            for device in devices:
                for location in device.sensor_data:
                    lock = get_lock_for_location(lock_per_location, location)
                    if lock is None:
                        add_lock_for_location(lock_per_location, location)

            self.barrier = ReusableBarrierSem(len(devices))
            self.taskscheduler = TaskScheduler(lock_per_location)

            
            for device in devices:
                device.share_taskscheduler(self.taskscheduler)
                device.share_barrier(self.barrier)

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        

        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        

        self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        while self.device.barrier is None:
            pass

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            
            if neighbours is None:
                if self.device.device_id == 0:
                    
                    self.device.taskscheduler.finish = True
                    self.device.taskscheduler.wait_workers()
                break

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            for (script, location) in self.device.scripts:
                new_task = Task(self.device, script, location, neighbours)
                self.device.taskscheduler.add_task(new_task)

            
            self.device.barrier.wait()


class Task(object):
    

    def __init__(self, device, script, location, neighbours):
        

        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def execute(self):
        

        script_data = []

        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

		
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)

            
            self.device.set_data(self.location, result)


from threading import Thread, Lock

class TaskScheduler(object):
    

    def __init__(self, lock_per_location):
        

        self.nr_threads = 16
        self.lock_per_location = lock_per_location
        self.workpool = []
        self.workpool_lock = Lock()
        self.workers_list = []
        self.finish = False

        self.start_workers()

    def add_task(self, new_task):
        

        with self.workpool_lock:
            self.workpool.append(new_task)

    def get_task(self):
        

        self.workpool_lock.acquire()
        if self.workpool != []:
            ret = self.workpool.pop()
        else:
            ret = None
        self.workpool_lock.release()
        return ret

    def start_workers(self):
        

        tid = 0
        while tid < self.nr_threads:
            thread = Worker(self)
            self.workers_list.append(thread)
            tid += 1

        for worker in self.workers_list:
            worker.start()

    def wait_workers(self):
        

        for worker in self.workers_list:
            worker.join()

    def get_lock_per_location(self, location):
        

        for (loc, lock) in self.lock_per_location:
            if loc == location:
                return lock
        return None


class Worker(Thread):
    

    def __init__(self, taskscheduler):
        

        Thread.__init__(self)
        self.taskscheduler = taskscheduler

    def run(self):
        

        while True:
            if self.taskscheduler.finish == True:
                break

            while True:
                task = self.taskscheduler.get_task()
                if task is None:
                    break
                with self.taskscheduler.get_lock_per_location(task.location):
                    task.execute()

