

from threading import Lock, Thread, Semaphore, Event
from Queue import Queue


class ReusableBarrier(object):
    

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
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None

        
        
        self.locks = {}
        for spot in sensor_data:
            self.locks[spot] = Lock()
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        for loc in self.sensor_data:
            if loc == location:
                self.locks[loc].acquire()
                return self.sensor_data[loc]

        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.dev_threads = ThreadsForEachDevice(8)

    def run(self):

        self.dev_threads.device = self.device

        while True:

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                self.dev_threads.jobs_to_be_done.put(
                    (neighbours, script, location))

            self.device.timepoint_done.clear()

            
            self.dev_threads.jobs_to_be_done.join()
            
            self.device.barrier.wait()

        
        self.dev_threads.jobs_to_be_done.join()

        for _ in range(len(self.dev_threads.threads)):
            self.dev_threads.jobs_to_be_done.put((None, None, None))

        for d_th in self.dev_threads.threads:
            d_th.join()


class ThreadsForEachDevice(object):
    

    def __init__(self, number_of_threads):
        self.device = None
        
        
        
        self.jobs_to_be_done = Queue(number_of_threads)
        self.threads = []

        self.create_threads(number_of_threads)
        self.start_threads()

    def create_threads(self, number_of_threads):
        
        for _ in range(number_of_threads):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

    def start_threads(self):
        
        for i_th in self.threads:
            i_th.start()

    def execute(self):
        
        while True:
            
            neighbours, script, location = self.jobs_to_be_done.get()
            
            if neighbours is None and script is None:
                self.jobs_to_be_done.task_done()
                return

            data_for_script = []
            
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        data_for_script.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                data_for_script.append(data)

            if data_for_script != []:
                
                scripted_data = script.run(data_for_script)

                
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, scripted_data)

                
                self.device.set_data(location, scripted_data)

            self.jobs_to_be_done.task_done()
