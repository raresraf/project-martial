


from threading import Event, Semaphore, Lock, Thread
from Queue import Queue


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
        self.location_lock = None
        self.NUM_THREADS = 8

    def __str__(self):
        
        return "Device %d" % self.device_id

    def is_master_thread(self, devices):
        
        for device in devices:
            if device.device_id < self.device_id:
                return 0
        return 1

    def setup_devices(self, devices):
        

        if self.is_master_thread(devices) == 1:
            barrier = ReusableBarrier(len(devices))
            location_lock = {}
            self.set_barrier_lock(devices, barrier, location_lock)

    def set_barrier_lock(self, devices, barrier, location_lock):
        
        for device in devices:
            device.barrier = barrier
            for location in device.sensor_data:
                if location not in location_lock:
                    location_lock[location] = Lock()
            device.location_lock = location_lock

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        

        data = None
        if location in self.sensor_data:
            data = self.sensor_data[location]


        return data

    def set_data(self, location, data, source=None):
        

        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_scripts(self, queue, neighbours):
        try:
            (script, location) = queue.get_nowait()
            lock_location = self.device.location_lock.get(location)
            lock_location.acquire()
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
            lock_location.release()
            queue.task_done()
        except:
            pass

    def start_threads(self, threadlist):
        for thread in threadlist:
            thread.start()

    def join_threads(self, threadlist):
        for thread in threadlist:
            thread.join()

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.device.timepoint_done.wait()

            queue = Queue()
            for (script, location) in self.device.scripts:
                queue.put((script, location))

            threadlist = []
            for thread in range(self.device.NUM_THREADS):
                thread = Thread(target=self.run_scripts, args=(queue, neighbours))
                threadlist.append(thread)
            self.start_threads(threadlist)
            self.join_threads(threadlist)
            queue.join()

            self.device.timepoint_done.clear()
            self.device.barrier.wait()

class ReusableBarrier():
    

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
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()
