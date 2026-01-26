


from threading import Event, Thread, Lock, Semaphore, Condition


class ReusableBarrierCond(object):
    

    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        
        self.cond = Condition()

    def wait(self):
        
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            
            self.cond.wait()
        
        self.cond.release()


class SignalType(object):
    
    SCRIPT_RECEIVED = 1
    TIMEPOINT_DONE = 2
    TERMINATION = 3


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        
        self.devices_barrier = None
        self.signal_received = Event()
        self.signal_type = None
        self.timepoint_work_done = Event()
        self.signal_sent = Event()
        self.data_locks = {}
        self.scripts_lock = Lock()

        
        self.thread = DeviceThread(self)
        self.thread.start()

        
        for location in sensor_data:
            self.data_locks[location] = Lock()

        self.devices_lock = {}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        if self.device_id == 0:
            devices_barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                device.devices_barrier = devices_barrier
                for location in device.sensor_data:
                    self.devices_lock[location] = Lock()

                device.devices_lock = self.devices_lock

    def assign_script(self, script, location):
        
        if script is not None:
            
            with self.scripts_lock:
                self.scripts.append((script, location))

            
            self.signal_type = SignalType.SCRIPT_RECEIVED
            self.signal_received.set()
            
            self.signal_sent.wait()
            self.signal_sent.clear()

        else:
            
            self.signal_type = SignalType.TIMEPOINT_DONE
            self.signal_received.set()
            
            self.signal_sent.wait()
            self.signal_sent.clear()
            
            self.timepoint_work_done.wait()
            self.timepoint_work_done.clear()

    def get_data(self, location):
        
        

        if location in self.sensor_data:
            with self.data_locks[location]:
                return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        
        
        if location in self.sensor_data:
            with self.data_locks[location]:
                self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        
        self.neighbours = None

        
        self.signal_received = Event()
        self.signal_type = None
        self.scripts_index = 0

        self.new_timepoint = Semaphore(0)
        self.signal_lock = Lock()

        
        self.num_threads = 8
        self.timepoint_computation_done = [Event() for _ in range(self.num_threads)]
        self.threads = [ComputationThread(self, count) for count in range(self.num_threads)]
        for count in range(self.num_threads):
            self.threads[count].start()

        self.neighbour_locks = [Lock() for _ in range(self.num_threads)]

    def acquire_neighbours(self):
        
        for lock in self.neighbour_locks:
            lock.acquire()

    def release_neighbours(self):
        
        for lock in self.neighbour_locks:
            lock.release()

    def run(self):
        
        
        while True:
            
            self.acquire_neighbours()
            
            self.neighbours = self.device.supervisor.get_neighbours()

            
            if self.neighbours is None:
                
                self.signal_type = SignalType.TERMINATION
                self.device.signal_sent.set()
                self.release_neighbours()

                
                for computation_thread_done in self.timepoint_computation_done:
                    computation_thread_done.wait()
                    computation_thread_done.clear()
                break

            self.release_neighbours()

            
            while True:
                
                self.device.signal_received.wait()
                self.device.signal_received.clear()

                
                self.signal_type = self.device.signal_type
                
                self.signal_received.set()
                self.device.signal_sent.set()

                
                if self.signal_type == SignalType.TIMEPOINT_DONE:
                    
                    for computation_thread_done in self.timepoint_computation_done:
                        computation_thread_done.wait()
                        computation_thread_done.clear()

                    
                    self.scripts_index = 0

                    
                    self.device.timepoint_work_done.set()
                    break

            
            self.device.devices_barrier.wait()

        
        for computation_thread in self.threads:
            computation_thread.join()


class ComputationThread(Thread):
    

    def __init__(self, device_thread, thread_id):
        
        Thread.__init__(self, name="Computing Thread %d" % thread_id)

        self.device_thread = device_thread
        self.thread_id = thread_id

    def run(self):
        

        


        
        while True:
            
            self.device_thread.signal_received.wait()
            
            self.device_thread.neighbour_locks[self.thread_id].acquire()

            
            if self.device_thread.signal_type == SignalType.TERMINATION:
                self.device_thread.neighbour_locks[self.thread_id].release()
                
                self.device_thread.timepoint_computation_done[self.thread_id].set()
                break

            
            while True:
                
                self.device_thread.device.scripts_lock.acquire()

                
                if len(self.device_thread.device.scripts) == self.device_thread.scripts_index:
                    
                    self.device_thread.device.scripts_lock.release()
                    
                    self.device_thread.timepoint_computation_done[self.thread_id].set()

                    
                    break

                
                index = self.device_thread.scripts_index
                (script_todo, location) = self.device_thread.device.scripts[index]
                self.device_thread.scripts_index += 1

                self.device_thread.device.scripts_lock.release()

                script_data = []
                
                for device in self.device_thread.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device_thread.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    
                    result = script_todo.run(script_data)

                    
                    for device in self.device_thread.neighbours:
                        device.set_data(location, result)

                    
                    self.device_thread.device.set_data(location, result)

            
            
            
            if self.device_thread.signal_type == SignalType.SCRIPT_RECEIVED:
                self.device_thread.signal_received.clear()

            self.device_thread.neighbour_locks[self.thread_id].release()
