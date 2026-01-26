




from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = None
        self.work_pool = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            lock_locations = []

            
            for device in devices:
                for _ in xrange(len(device.sensor_data)):
                    lock_locations.append(Lock())

            
            for device in devices:
                tasks_finish = Event()
                device.work_pool = WorkPool(tasks_finish, lock_locations)

                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data \
                else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class WorkPool(object):
    



    def __init__(self, tasks_finish, lock_locations):
        

        self.workers = []
        self.tasks = []
        self.current_task_index = 0
        self.lock_get_task = Lock()
        self.work_to_do = Event()
        self.tasks_finish = tasks_finish
        self.lock_locations = lock_locations
        self.max_num_workers = 8

        for i in xrange(self.max_num_workers):
            self.workers.append(Worker(self, self.lock_get_task, \
                    self.work_to_do, self.lock_locations))
            self.workers[i].start()

    def set_tasks(self, tasks):
        
        self.tasks = tasks
        self.current_task_index = 0

        self.work_to_do.set()

    def get_task(self):
        
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index]

            self.current_task_index = self.current_task_index + 1

            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear()
                self.tasks_finish.set()

            return task
        else:
            return None

    def close(self):
        
        self.tasks = []
        self.current_task_index = len(self.tasks)

        
        self.work_to_do.set()

        for worker in self.workers:
            worker.join()

class Worker(Thread):
    

    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        

        Thread.__init__(self)
        self.work_pool = work_pool
        self.lock_get_task = lock_get_task
        self.work_to_do = work_to_do
        self.lock_locations = lock_locations

    def run(self):
        while True:
            
            self.lock_get_task.acquire()

            
            self.work_to_do.wait()

            task = self.work_pool.get_task()

            
            self.lock_get_task.release()

            if task is None:
                break

            
            script = task[0]
            location = task[1]
            neighbours = task[2]
            self_device = task[3]

            
            self.lock_locations[location].acquire()

            script_data = []

            
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self_device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                for device in neighbours:
                    device.set_data(location, result)
                
                self_device.set_data(location, result)

            
            self.lock_locations[location].release()





class DeviceThread(Thread):
    

    def __init__(self, device, barrier, tasks_finish):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.tasks_finish = tasks_finish

    def run(self):
        while True:
            
            self.barrier.wait()

            
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                self.device.work_pool.close()
                break

            
            self.device.script_received.wait()

            
            tasks = []

            for (script, location) in self.device.scripts:
                tasks.append((script, location, neighbours, self.device))

            if tasks != []:
                
                self.device.work_pool.set_tasks(tasks)

                
                self.tasks_finish.wait()

                
                self.tasks_finish.clear()

            
            self.device.script_received.clear()
