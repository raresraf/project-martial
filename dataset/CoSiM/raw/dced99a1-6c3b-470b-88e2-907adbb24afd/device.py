


from Queue import Queue
from threading import Semaphore, Lock
from threading import Event, Thread


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.read_data = sensor_data
        self.supervisor = supervisor
        self.active_queue = Queue()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.time = 0

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id == 0:
            self.new_round = ReusableBarrierSem(len(devices))
            self.devices = devices
            for device in self.devices:
                device.new_round = self.new_round
        self.thread.start()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            for (script, location) in self.scripts:
                self.active_queue.put((script, location))
            for x in range(8):
                self.active_queue.put((-1, -1))

    def get_data(self, location):
        
        return self.read_data[location] if location in self.read_data else None

    def set_data(self, location, data):
        
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers_number = 8

    def run(self):
        neighbours = self.device.supervisor.get_neighbours()
        while True:
            self.workers = []
            self.device.neighbours = neighbours
            if neighbours is None:
                break

            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            for worker in self.workers:
                worker.join()
            self.device.new_round.wait()
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):

    def __init__(self, device):
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            script, location = self.device.active_queue.get()
            if script == -1:
                break
            script_data = []
            matches = []
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    matches.append(device)
                    script_data.append(data)
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
                matches.append(self.device)

            if len(script_data) > 1:
                result = script.run(script_data)
                for device in matches:
                    old_value = device.get_data(location)
                    if old_value < result:
                        device.set_data(location, result)


class ReusableBarrierSem():
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
