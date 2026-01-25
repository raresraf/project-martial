


from threading import Thread, Condition, Event, Lock
from Queue import Queue

def stop():
    
    return

class MyBarrier(object):
    
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
    def wait(self):
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class MyWorkerThread(Thread):
    
    def __init__(self, tasks_list, lock):
        
        Thread.__init__(self)
        self.tasks_list = tasks_list
        self.daemon = True
        self.stop = False
        self.lock = lock
        self.start()

    def run(self):
        while True:
            function, params = self.tasks_list.get()
            
            
            if function is stop:
                self.tasks_list.task_done()
                self.lock.release()
                break

            
            function(*params)

            self.tasks_list.task_done()

class MyThreadPool(object):
    
    def __init__(self, no_threads):
        
        self.no_threads = no_threads
        self.tasks_list = Queue(no_threads)
        self.worker_list = []
        self.lock = Lock()
        
        for _ in xrange(no_threads):
            self.worker_list.append(MyWorkerThread(self.tasks_list, self.lock))

    def add(self, function, *params):
        
        self.tasks_list.put((function, params))

    def wait(self):
        
        self.tasks_list.join()
        for i in xrange(self.no_threads):
            self.lock.acquire()
            
            self.add(stop, None)
        for i in xrange(self.no_threads):


            self.worker_list[i].join()



class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.barrier = None
        
        self.locks = []
        self.no_locations = self.supervisor.supervisor.testcase.num_locations
        
        self.pool = MyThreadPool(8)

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        if self.device_id == 0:
            barrier = MyBarrier(len(devices))

            for i in xrange(self.no_locations):
                self.locks.append(Lock())

            for i in xrange(len(devices)):
                devices[i].barrier = barrier
                devices[i].locks = self.locks


    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location \
                in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_script(self, params):
        
        neighbours, script, location = params
        self.device.locks[location].acquire()

        script_data = []
        
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data)
            
            
            for device in neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)



        self.device.locks[location].release()

    def run(self):
        
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                self.device.pool.wait()
                break

            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                params = (neighbours, script, location)
                
                self.device.pool.add(self.run_script, params)
            
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
