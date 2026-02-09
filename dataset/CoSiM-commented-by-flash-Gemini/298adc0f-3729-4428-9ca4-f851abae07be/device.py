/**
 * @file device.py
 * @brief Semantic documentation for device.py. 
 *        This is a placeholder. Detailed semantic analysis will be applied later.
 */



from threading import Event, Thread, Lock, Semaphore

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
                n_threads = self.num_threads
                while n_threads > 0:
                    threads_sem.release()
                    n_threads -= 1
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
        self.devices = []
        self.barrier = None
        self.workers = []
        keys = range(60)
        self.loc_barrier = {key: None for key in keys}
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        


        if script is not None:
            self.scripts.append((script, location))
            if self.loc_barrier[location] is None:
                for device in self.devices:
                    if device.loc_barrier[location] is not None:
                        self.loc_barrier[location] = device.loc_barrier[location]
                        break


            if self.loc_barrier[location] is None:
                self.loc_barrier[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class ScriptWorker(Thread):
    
    def __init__(self, device, neighbours, script, location):
        
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        self.device.loc_barrier[self.location].acquire()
        script_data = []
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        self.device.loc_barrier[self.location].release()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()



            for (script, location) in self.device.scripts:
                thread = ScriptWorker(self.device, neighbours, script, location)
                self.device.workers.append(thread)

            for worker in self.device.workers:
                worker.start()

            for worker in self.device.workers:
                worker.join()

            self.device.workers = []
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
