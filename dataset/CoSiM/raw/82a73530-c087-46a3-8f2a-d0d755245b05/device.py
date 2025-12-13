


from threading import Lock, Semaphore, Thread, Event
from Queue import Queue

class RBarrier(object):
    
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
        
        self.location_lock = {}
        self.barrier = None
        self.all_devices = []
        self.update = Lock()
        self.got_data = False
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        self.all_devices = devices
        
        
        if self.device_id == 0:
            self.barrier = RBarrier(len(self.all_devices))
            self.update = Lock()
            for i in xrange(0, len(self.all_devices)):
                self.all_devices[i].barrier = self.barrier
                self.all_devices[i].update = self.update

    def assign_script(self, script, location):
        
        if script is not None:
            if location not in self.location_lock:
                self.got_data = True
                self.location_lock[location] = Lock()
                
                with self.update:
                    for i in xrange(0, len(self.all_devices)):
                        self.all_devices[i].location_lock[
                            location] = self.location_lock[location]

            if location in self.location_lock:
                self.scripts.append((script, location))
                self.script_received.set()
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
        
        self.thread.join()


class DeviceThread(Thread):
    
    THREAD_NUMBER = 8
    STOP_FLAG = "STOP"

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_number = DeviceThread.THREAD_NUMBER
        self.queue = Queue(self.thread_number)
        self.threads = [Thread(target=self.thread_func) for _ in [None]*self.thread_number]
        _ = [x.start() for x in self.threads]
        self.debug_timepoint = 0
        self.transmit = 0

    def run(self):
        
        while True:


            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            while True:
                self.device.timepoint_done.wait()
                if not self.device.got_data:
                    self.device.timepoint_done.clear()
                    self.transmit += 1
                    self.device.got_data = True
                    break
                else:
                    _ = [self.queue.put((neighbours, x[0], x[1])) \
                     for x in self.device.scripts]
                    self.device.got_data = False

            self.debug_timepoint += 1
            self.queue.join()
            self.device.barrier.wait()

        self.queue.join()
        _ = [self.queue.put((
            DeviceThread.STOP_FLAG, \
            DeviceThread.STOP_FLAG, \
            DeviceThread.STOP_FLAG)) \
            for _ in xrange(self.THREAD_NUMBER)]
        _ = [x.join() for x in self.threads]

    def thread_func(self):
        
        neighbours, script, location = self.queue.get()
        while neighbours is not DeviceThread.STOP_FLAG:
            script_data = []
            with self.device.location_lock[location]:
                for device in neighbours:
                    if device.device_id != self.device.device_id:
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
            self.queue.task_done()
            neighbours, script, location = self.queue.get()
        self.queue.task_done()
