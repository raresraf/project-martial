


from threading import Event, Semaphore, Lock, Thread
from Queue import Queue


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.all_scripts_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        TOTAL_THREADS = 32

        
        self.NUM_THREADS = TOTAL_THREADS / len(devices) + 1
        
        
        lower = 0
        for device in devices:
            if device.device_id < self.device_id:
                lower += 1
        if lower < TOTAL_THREADS % len(devices):
            self.NUM_THREADS += 1

        if lower == 0:
            
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.on_time_point_barrier(barrier)
            
            location_lock = {}
            for device in devices:
                for location in device.sensor_data:
                    if location not in location_lock:
                        location_lock[location] = Lock()
            for device in devices:
                device.on_location_lock_dictionary(location_lock)

    def on_time_point_barrier(self, barrier):
        
        self.barrier = barrier

    def on_location_lock_dictionary(self, location_lock):
        
        self.location_lock = location_lock

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.all_scripts_received.set()

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

    def run(self):

        while True:
            
            self.device.all_scripts_received.clear()
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.all_scripts_received.wait()

            
            q = Queue()
            for job in self.device.scripts:
                q.put(job)

            
            for t in range(self.device.NUM_THREADS):
                
                runner = ScriptRunner(q, neighbours, self.device)
                runner.start()
            


            q.join()
            
            self.device.barrier.wait()

class ScriptRunner(Thread):
    

    def __init__(self, queue, neighbours, device):
        Thread.__init__(self)
        self.queue = queue
        self.neighbours = neighbours
        self.device = device

    def run(self):
        
        try:
            (script, location) = self.queue.get_nowait()
            
            self.device.location_lock[location].acquire()

            script_data = []
            
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)
                
                for device in self.neighbours:


                    device.set_data(location, result, self.device.device_id)
                
                self.device.set_data(location, result)

            
            self.device.location_lock[location].release()
            
            self.queue.task_done()
        except:
            pass

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
