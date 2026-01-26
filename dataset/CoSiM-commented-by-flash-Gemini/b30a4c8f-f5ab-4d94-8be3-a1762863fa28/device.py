


from threading import Event, Thread, Semaphore, Lock


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.locks = {}
        for x in sensor_data.keys():
            self.locks[x] = Lock()
        self.supervisor = supervisor
        self.script_received = Semaphore(0)
        self.scripts = []
        self.scripts_number = 0
        self.canal = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.time = 0

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        self.devices = devices
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in self.devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            for a in self.scripts:
                self.scripts_number += 1
                self.canal.insert(0, a)
                self.script_received.release()
            self.timepoint_done.set()

    def get_data(self, location):
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads_number = 16
        self.tasks_done = Semaphore(0)
        self.stop = False

        self.threads = [Thread(target=self.script_work, args=(i,))
                        for i in range(self.threads_number)]

    def start_threads(self):
        
        for thread in self.threads:
            thread.start()

    def stop_threads(self):
        
        for thread in self.threads:
            thread.join()

    def script_work(self, id):
        
        while True:
            self.device.script_received.acquire()
            script, location = self.device.canal.pop()
            if script is None:
                break

            script_data = []
            all_data = [(device.get_data(location), device)
                        for device in self.device.neighbours
                        if device.get_data(location)]

            data = self.device.get_data(location)
            if data is not None:
                all_data.append((data, self.device))
            script_data = [x for x, _ in all_data]
            neighbours = [x for _, x in all_data]

            if len(script_data) > 1:
                result = script.run(script_data)
                for neighbour in neighbours:
                        with neighbour.locks[location]:
                            data = neighbour.get_data(location)
                            if data < result:
                                neighbour.set_data(location, result)

            self.tasks_done.release()

    def run(self):
        
        self.device.neighbours = self.device.supervisor.get_neighbours()
        self.start_threads()
        self.device.barrier.wait()

        while True:
            if self.device.neighbours is None:
                self.stop = True
                for i in range(self.threads_number):
                    self.device.canal.insert(0, (None, None))
                    self.device.script_received.release()
                break

            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            for i in range(self.device.scripts_number):
                self.tasks_done.acquire()
            self.device.scripts_number = 0
            self.device.barrier.wait()
            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.time += 1

        self.stop_threads()


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
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
