


from threading import Event, Thread, Lock, Semaphore, Condition
from Queue import Queue
from barrier import ReusableBarrierSem

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor    
        self.scripts = []
        self.all_devices = []

        
        self.event_timepoint_done = Event()

        
        self.event_setup_done = Event()

        
        self.event_stop_threads = Event()

        
        self.lock_data = Lock()

        
        self.lock_locations = []

        
        self.queue_scripts = Queue()

        
        
        self.semaphore_queue = Semaphore(0)

        
        self.barrier_devices = None

        
        self.condition_variable = Condition(Lock())

        self.thread = DeviceThread(self)
        self.thread.start()


    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        

        self.all_devices = devices

        
        if self.device_id == 0:
            
            nr_of_devices = devices.__len__()
            barrier_devices = ReusableBarrierSem(nr_of_devices)

            
            for _ in range(24):
                self.lock_locations.append(Lock())

            for device in devices:
                device.barrier_devices = barrier_devices
                device.lock_locations = self.lock_locations
                device.event_setup_done.set()

    def assign_script(self, script, location):
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            
            self.event_timepoint_done.set()

    def get_data(self, location):
        

        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        

        
        with self.lock_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        self.device.event_setup_done.wait()

        
        worker_threads = []
        for thread_id in range(8):
            thread = WorkerThread(self.device, thread_id)
            worker_threads.append(thread)
            worker_threads[-1].start()

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            
            if neighbours is None:
                break

            
            self.device.event_timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                thread_script = (script, location, neighbours)

                self.device.queue_scripts.put(thread_script)

                
                self.device.semaphore_queue.release()

            self.device.event_timepoint_done.clear()

            
            self.device.condition_variable.acquire()
            while self.device.queue_scripts.empty() is False:
                self.device.condition_variable.wait()
            self.device.condition_variable.release()

            
            


            self.device.barrier_devices.wait()

        self.device.event_stop_threads.set()

        
        for _ in range(8):
            self.device.semaphore_queue.release()

        for thread in worker_threads:
            thread.join()



class WorkerThread(Thread):
    

    def __init__(self, device, my_id):
        

        Thread.__init__(self, name="Worker Thread %d" % my_id)

        self.device = device

    def run(self):
        while True:
            
            self.device.semaphore_queue.acquire()



            if self.device.event_stop_threads.is_set():
                break

            thread_script = self.device.queue_scripts.get()

            script = thread_script[0]
            location = thread_script[1]
            neighbours = thread_script[2]

            

            
            with self.device.lock_locations[location]:
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

            
            self.device.condition_variable.acquire()
            self.device.condition_variable.notify()
            self.device.condition_variable.release()
