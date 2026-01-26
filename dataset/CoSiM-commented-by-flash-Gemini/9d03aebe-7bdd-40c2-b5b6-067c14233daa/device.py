


from threading import Event, Thread, Lock

from barrier import ReusableBarrierSem
from threadpool import ThreadPoll, comparator



class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.script_received.set()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.lock = Event()
        self.lock.clear()
        self.datalock = Lock()
        self.personal_lock = []
        self.bariera = None
        self.all = None
        self.no_devices = None

        
        crt = max(self.sensor_data.keys())
        for _ in xrange(crt + 1):
            self.personal_lock.append(Lock())


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        

        
        self.all = sorted(devices, cmp=comparator)
        self.no_devices = len(self.all)

        
        if self.device_id == 0:
            self.bariera = ReusableBarrierSem(len(self.all))
            self.lock.set()

        else:
            
            prev_device = self.all[self.device_id - 1]
            prev_device.lock.wait()
            self.bariera = prev_device.bariera
            self.lock.set()

        
        self.thread.start()



    def assign_script(self, script, location):
        
        
        if script is not None:
            self.script_received.wait()
            self.scripts.append((script, location))
        else:
            self.script_received.clear()
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):

        
        thread_pool = ThreadPoll(8)
        while True:

            self.device.bariera.wait()

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                thread_pool.close()
                break

            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            work_list = []

            
            for (script, location) in self.device.scripts:
                work_list.append((location, script, self.device, neighbours))

            
            thread_pool.put_work(work_list)

            self.device.script_received.set()

from threading import Event, Thread, Semaphore

def comparator(device_a, device_b):
    
    if device_a.device_id > device_b.device_id:
        return 1
    else:
        return -1



class Solve(Thread):
    
    def __init__(self, sem, free_threads, working_threads):
        
        Thread.__init__(self)

        self.free_threads = free_threads
        self.sem = sem
        self.working_threads = working_threads
        self.work = Event()
        self.free = Event()
        self.done = 0
        self.work.clear()
        self.free.set()

        self.location = None
        self.script = None
        self.device = None
        self.neighbours = None


    def set_work(self, location, script, device, neighbours):
        
        self.location = location
        self.script = script
        self.device = device
        self.neighbours = neighbours


    def run(self):
        
        while 1:

            
            self.work.wait()

            
            if self.done == 1:
                break

            script_data = []

            
            list_neighbours = self.neighbours
            list_neighbours.append(self.device)
            list_neighbours = set(list_neighbours)
            list_neighbours = sorted(list_neighbours, cmp=comparator)

            
            for device in list_neighbours:
                if self.location in device.sensor_data:
                    device.personal_lock[self.location].acquire()

            
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)

            
            for device in reversed(list_neighbours):
                if self.location in device.sensor_data:
                    device.personal_lock[self.location].release()

            
            self.free.set()
            self.work.clear()
            self.free_threads.append(self)
            
            self.sem.release()



class ThreadPoll(object):
    
    def __init__(self, no_threads):
        
        self.no_threads = no_threads
        self.free_threads = []
        self.working_threads = []
        self.all_threads = []
        self.workdone = Event()
        self.sem = Semaphore(self.no_threads)

        
        for _ in xrange(0, no_threads):
            tmp = Solve(self.sem, self.free_threads, self.working_threads)
            self.free_threads.append(tmp)
            self.all_threads.append(tmp)

        for current_thread in self.free_threads:
            current_thread.start()

    def put_work(self, work_list):
        
        
        for (location, script, device, neighbours) in work_list:
            self.sem.acquire()
            current_thread = self.free_threads.pop(0)
            current_thread.set_work(location, script, device, neighbours)
            current_thread.free.clear()
            current_thread.work.set()

        
        for current_thread in self.all_threads:
            current_thread.free.wait()

    def close(self):
        
        for current_thread in self.all_threads:
            current_thread.done = 1
            current_thread.work.set()

        
        for current_thread in self.all_threads:
            current_thread.join()

