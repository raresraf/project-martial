


from threading import Event, Thread, Lock, Condition

class ReusableBarrier(object):
    
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
        self.thread_number = 8
        self.thread_lock = Lock()
        self.thread_barrier = ReusableBarrier(self.thread_number)
        self.neighbours_list = []
        self.neighbours_list_collected = False
        self.scripts = []
        self.locations_list = {}
        self.global_barrier = None
        self.setup_event = Event()
        self.script_received = Event()
        self.timepoint_done = Event()
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.threads = []
        for i in range(0, self.thread_number):
            self.threads.append(DeviceThread(self, i))
        for i in range(0, self.thread_number):
            self.threads[i].start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        if self.device_id == 0:
            
            for device in devices:
                for loc in device.sensor_data:
                    self.locations_list.update({loc:Lock()})
            
            max_threads = self.thread_number * len(devices)
            self.global_barrier = ReusableBarrier(max_threads)
            self.setup_event.set()
        
        else:
            
            main_device = None
            for dev in devices:
                if dev.device_id == 0:
                    main_device = dev
            main_device.setup_event.wait()
            self.global_barrier = main_device.global_barrier
            self.locations_list = main_device.locations_list
            self.setup_event.set()

    def assign_script(self, script, location):
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        for i in range(0, self.thread_number):
            self.threads[i].join()


class DeviceThread(Thread):
    

    def __init__(self, device, my_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.my_id = my_id

    def run(self):
        
        self.device.setup_event.wait()


        while True:
            
            self.device.thread_lock.acquire()
            if self.device.neighbours_list_collected is False:
                self.device.neighbours_list_collected = True
                self.device.neighbours_list = self.device.supervisor.get_neighbours()
            self.device.thread_lock.release()



            if self.device.neighbours_list is None:
                break

            self.device.timepoint_done.wait()

            
            for i in xrange(self.my_id, len(self.device.scripts), self.device.thread_number):
                
                (script, location) = self.device.scripts[i]
                script_data = []
                
                self.device.locations_list[location].acquire()

                
                for device in self.device.neighbours_list:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in self.device.neighbours_list:
                        device.set_data(location, result)
                    


                    self.device.set_data(location, result)
                
                self.device.locations_list[location].release()

            
            self.device.global_barrier.wait()

            
            self.device.thread_lock.acquire()
            if self.device.neighbours_list_collected is True:
                self.device.neighbours_list_collected = False
                self.device.timepoint_done.clear()
            self.device.thread_lock.release()
            self.device.thread_barrier.wait()
