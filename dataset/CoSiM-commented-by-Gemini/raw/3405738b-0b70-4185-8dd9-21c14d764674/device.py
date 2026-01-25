


from threading import Lock, Thread, Event
from Queue import Queue
from barrier import ReusableBarrier


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = list()
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.lock_dict = None

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        
        
        
        if self.device_id is 0:
            barrier = ReusableBarrier(len(devices))
            lock_dict = dict()

            for device in devices:
                device.barrier = barrier
                device.lock_dict = lock_dict

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
        
        self.thread.join()


class DeviceThread(Thread):
    

    def __init__(self, device):
        

        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue()
        self.thread_pool = list()

        self.thread_num = 8
        
        for _ in range(0, self.thread_num):
            my_thread = Thread(target=self.executor_service)
            my_thread.start()
            self.thread_pool.append(my_thread)

    def run(self):

        while "Not finished":
            
            neighbours = self.device.supervisor.get_neighbours()

            
            if neighbours is None:
                for _ in range(0, self.thread_num):
                    self.queue.put(None)
                self.shutdown()
                self.thread_pool = list()
                break

            
            self.device.timepoint_done.wait()

            
            
            for (script, location) in self.device.scripts:
                queue_info = [script, location, neighbours]


                self.queue.put(queue_info)

            
            self.queue.join()

            
            self.device.barrier.wait()

            
            self.device.timepoint_done.clear()


    def executor_service(self):
        

        while "Not finished":
            
            tasks = self.queue.get()

            
            if tasks is None:
                self.queue.task_done()
                break
            else:
                
                script_t = tasks[0]
                location_t = tasks[1]
                neighbours_t = tasks[2]

            
            
            if self.device.lock_dict.get(location_t) is None:
                self.device.lock_dict[location_t] = Lock()

            
            self.device.lock_dict[location_t].acquire()

            
            
            self.data_processing(self.device, script_t, location_t, neighbours_t)

            
            self.device.lock_dict[location_t].release()

            
            self.queue.task_done()

    @classmethod
    def data_processing(cls, device, script, location, neighbours):
        

        
        script_info = list()
        for i in range(0, len(neighbours)):
            data = neighbours[i].get_data(location)
            if data:
                script_info.append(data)

        data = device.get_data(location)
        
        if data != None:
            script_info.append(data)

        if script_info:
            
            result = script.run(script_info)
            
            send_info = [location, result]

            
            for i in range(0, len(neighbours)):
                neighbours[i].set_data(send_info[0], send_info[1])

            
            device.set_data(send_info[0], send_info[1])




    def shutdown(self):
        

        for i in range(0, self.thread_num):
            self.thread_pool[i].join()
