


from threading import Event, Thread
from threading import Lock, Semaphore

class ReusableBarrier():

    
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        
        with self.count_lock:
            count_threads[0] -= 1

            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.done_setup = Event()
        self.device_id = device_id
        self.thread = DeviceThread(self)
        self.thread.start()
        self.script_received = Event()
        self.sensor_data = sensor_data
        self.semaphore = Semaphore(value=8)

        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        self.nr_thread = 0
        self.lock_timepoint = Lock()
        self.script_list = []
        self.lock_index = []

        self.r_barrier = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        used_devices = len(devices)
        if self.device_id is 0:
            r_barrier = ReusableBarrier(used_devices)
            for _ in range(0, 24):
                self.lock_index.append(Lock())

            for d in range(len(devices)):
                devices[d].lock_index = self.lock_index
                devices[d].r_barrier = r_barrier
                devices[d].done_setup.set()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def close_scripts(self):
        
        nrThreads = len(self.script_list)
        for i in range(0, nrThreads):
            self.script_list[i].join()

        for i in range(0, nrThreads):
            self.script_list.pop()

        self.nr_thread = 0

    def shutdown(self):
        
        self.thread.join()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_script(self, neighbours):
        
        for (script, location) in self.device.scripts:
            self.device.semaphore.acquire()
            self.device.script_list.append(DataScript\
            (neighbours, location, script, self.device))

            self.device.nr_thread = self.device.nr_thread + 1
            self.device.script_list[len(self.device.script_list)-1].start()

    def run(self):
        

        
        self.device.done_setup.wait()

        while True:
            
            

            with self.device.lock_timepoint as neighbours:


                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    break

            
            self.device.timepoint_done.wait()
            self.run_script(neighbours)

            self.device.r_barrier.wait()
            self.device.timepoint_done.clear()
            self.device.close_scripts()
            self.device.r_barrier.wait()


class DataScript(Thread):
    
    def __init__(self, neighbours, location, script, scr_device):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.scr_device = scr_device


    def getdata(self, script_data):
        
        data = self.scr_device.get_data(self.location)
        if data is not None:
            script_data.append(data)

    def scriptdata(self, script_data):
        
        if script_data != []:
            
            result = self.script.run(script_data)
            
            for device in self.neighbours:
                device.set_data(self.location, result)

            self.scr_device.set_data(self.location, result)
        self.scr_device.semaphore.release()

    def run(self):
        with self.scr_device.lock_index[self.location]:
            script_data = []

            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            self.getdata(script_data)
            self.scriptdata(script_data)
