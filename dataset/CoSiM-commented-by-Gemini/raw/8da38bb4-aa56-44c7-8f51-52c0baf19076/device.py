


from threading import Event, Lock, Thread, RLock, Semaphore

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
        self.lock = RLock()
        self.script_lock = RLock()
        self.run_lock = RLock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.device_id is 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        
        with self.script_lock:
            if script is not None:
                self.scripts.append((script, location))
                self.script_received.set()
            else:
                self.timepoint_done.set()

    def get_data(self, location):
        
        with self.lock:
            result = self.sensor_data[location] if location in self.sensor_data else None
        return result

    def set_data(self, location, data):
        

        with self.lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            neighbours.append(self.device)

            self.device.timepoint_done.wait()

            
            num_threads = 8
            threads = [Thread(target=self.concurrent_work,
                              args=(neighbours, i, num_threads)) for i in range(num_threads)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            self.device.barrier.wait()
            self.device.timepoint_done.clear()

    def concurrent_work(self, neighbours, thread_id, num_threads):
        
        for (script, location) in self.keep_assigned(self.device.scripts, thread_id, num_threads):
            script_data = []
            
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                for device in neighbours:
                    res = max(result, device.get_data(location))
                    device.set_data(location, res)

    def keep_assigned(self, scripts, thread_id, num_threads):
        
        assigned_scripts = []
        for i, script in enumerate(scripts):
            if i % num_threads is thread_id:
                assigned_scripts.append(script)

        return assigned_scripts
