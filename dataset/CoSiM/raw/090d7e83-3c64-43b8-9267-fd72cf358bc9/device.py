


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

class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.loc_lock = []
        self.crt_nb_scripts = 0
        self.crt_script = 0
        
        self.script_sem = Semaphore(value=1)
        
        self.done_processing = Semaphore(value=0)
        
        self.wait_for_next_timepoint = Event()
        self.wait_for_next_timepoint.set()
        
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        


        my_barrier = ReusableBarrierCond(len(devices))
        if self.device_id == 0:
            
            for i in range(0, len(devices)):
                devices[i].barr = my_barrier
            
            for i in range(0, 100):
                custom_lock = Lock()
                self.loc_lock.append(custom_lock)
            for devs in devices:
                if devs.device_id != 0:
                    devs.loc_lock = self.loc_lock
        else:
            
            while not hasattr(self, 'barr'):
                continue

    def assign_script(self, script, location):
        
        
        self.wait_for_next_timepoint.wait()
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
            
            self.crt_nb_scripts += 1
        else:
            
            
            self.wait_for_next_timepoint.clear()
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()



class MyWorker(Thread):
    
    def __init__(self, device, worker_barrier, my_id):
        Thread.__init__(self, name="Worker Thread %d" % my_id)
        self.my_dev = device
        self.worker_bar = worker_barrier
        self.my_id = my_id

    def run(self):
        while True:
            
            self.my_dev.device.timepoint_done.wait()
            neighbours = self.my_dev.neighbours
            if neighbours is None:
                break
            
            if self.my_dev.inner_state == 1:
                
                self.worker_bar.wait()
                
                if self.my_id == 0:
                    
                    
                    self.my_dev.device.timepoint_done.clear()
                    
                    self.my_dev.device.done_processing.release()
                    
                    self.my_dev.inner_state = 0
                
                self.worker_bar.wait()
                continue
            while True:
                
                self.my_dev.device.script_sem.acquire()
                if self.my_dev.device.crt_script == self.my_dev.device.crt_nb_scripts:
                    self.my_dev.device.script_sem.release()
                    if self.my_id == 0:
                        
                        self.my_dev.inner_state = 1
                    break
                
                my_script = self.my_dev.device.scripts[self.my_dev.device.crt_script]
                
                self.my_dev.device.crt_script += 1
                self.my_dev.device.script_sem.release()
                
                self.my_dev.device.loc_lock[my_script[1]].acquire()

                
                script_data = []
                
                for device in neighbours:


                    data = device.get_data(my_script[1])
                    if data is not None:
                        script_data.append(data)
                
                data = self.my_dev.device.get_data(my_script[1])
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = my_script[0].run(script_data)

                    
                    for device in neighbours:
                        device.set_data(my_script[1], result)


                    
                    self.my_dev.device.set_data(my_script[1], result)

                
                self.my_dev.device.loc_lock[my_script[1]].release()




class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.inner_state = 0
        
        self.thread_p = []
        
        self.w_bar = ReusableBarrierCond(8)
        
        self.neighbours = []
        self.inner_state_lock = Lock()

        for i in range(0, 8):
            self.thread_p.append(MyWorker(self, self.w_bar, i))

        
        for i in range(0, 8):
            self.thread_p[i].start()

    def run(self):
        
        while True:

            
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                
                self.device.timepoint_done.set()
                break
            
            self.device.done_processing.acquire()
            
            self.device.crt_script = 0
            
            self.device.wait_for_next_timepoint.set()
            
            self.device.barr.wait()

        for i in range(0, 8):
            self.thread_p[i].join()
