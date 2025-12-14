




from threading import Event, Thread, Lock, Condition, Semaphore


class Device(object):
    

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.devices = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        
        self.lead_device_index = -1

        self.location_locks = []


        if device_id == 0:
            
            
            self.threads_that_finished_no = 0

            
            self.next_time_point_cond = Condition()

            
            
            self.can_start = Event()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        

        self.devices = devices

        
        for i in xrange(len(self.devices)):
            if devices[i].device_id == 0:
                self.lead_device_index = i
                break

        if self.device_id == 0:
            self.can_start.clear()

            
            max_lock = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > max_lock:
                        max_lock = location

            for _ in range(0, max_lock + 1):
                self.location_locks.append(Lock())

            for device in devices:
                device.location_locks = self.location_locks

            self.can_start.set()
        else:
            devices[self.lead_device_index].can_start.wait()

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

    def notify_finish(self):
        
        self.devices[self.lead_device_index].next_time_point_cond.acquire()
        self.devices[self.lead_device_index].threads_that_finished_no += 1

        if self.devices[self.lead_device_index].threads_that_finished_no == len(self.devices):
            self.devices[self.lead_device_index].threads_that_finished_no = 0
            self.devices[self.lead_device_index].next_time_point_cond.notifyAll()
        else:
            self.devices[self.lead_device_index].next_time_point_cond.wait()

        self.devices[self.lead_device_index].next_time_point_cond.release()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = WorkerThreadPool(device)

    def run(self):
        
        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                self.thread_pool.shutdown()
                break

            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            for (script, location) in self.device.scripts:
                self.thread_pool.do_work(script, location, neighbours)

            self.thread_pool.wait_to_finish_work()

            self.device.notify_finish()

class WorkerThreadPool(object):
    
    def __init__(self, device):
        
        self.device = device
        self.work_finished_event = Event()
        self.work_finished_event.set()
        self.worker_pool = []

        self.ready_for_work_queue = []
        self.read_to_work_thread_sem = Semaphore(8)
        self.queue_lock = Lock()
        
        for _ in xrange(8):
            thread = SimpleWorker(self, self.device)
            self.worker_pool.append(thread)
            self.ready_for_work_queue.append(thread)
            thread.start()

    def do_work(self, script, location, neighbours):
        
        if self.work_finished_event.isSet():
            self.work_finished_event.clear()
        self.read_to_work_thread_sem.acquire()
        self.queue_lock.acquire()
        worker = self.ready_for_work_queue.pop(0)
        self.queue_lock.release()
        worker.do_work(script, location, neighbours)

    def shutdown(self):
        
        for worker in self.worker_pool:
            worker.should_i_stop = True
            worker.data_for_work_ready.release()
        for worker in self.worker_pool:
            worker.join()

    def worker_finished(self, worker):
        
        self.queue_lock.acquire()
        self.ready_for_work_queue.append(worker)

        if len(self.ready_for_work_queue) == 8 and not self.work_finished_event.isSet():
            self.work_finished_event.set()

        self.queue_lock.release()
        self.read_to_work_thread_sem.release()

    def wait_to_finish_work(self):
        
        self.work_finished_event.wait()



class SimpleWorker(Thread):
    
    def __init__(self, worker_pool, device):
        
        Thread.__init__(self)
        self.worker_pool = worker_pool
        self.should_i_stop = False
        self.data_for_work_ready = Semaphore(0)
        self.device = device
        self.script = None
        self.location = None
        self.neighbours = None

    def do_work(self, script, location, neighbours):
        
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.data_for_work_ready.release()


    def run(self):
        while True:
            self.data_for_work_ready.acquire()

            if self.should_i_stop is True:
                break
            self.device.location_locks[self.location].acquire()
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
            self.device.location_locks[self.location].release()
            self.worker_pool.worker_finished(self)
