


from threading import Thread, Event, Lock, Condition, Semaphore




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
                for _ in range(self.num_threads):
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
        self.loc_locks = []  
        self.condition = Condition()  
        self.barrier = None  
        self.neighbours = None
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            self.loc_locks = [None] * 100  
            for device in devices:  
                device.barrier = self.barrier
                device.loc_locks = self.loc_locks

    def assign_script(self, script, location):
        if script is not None:
            self.scripts.append((script, location))
            if self.loc_locks[location] is None:
                self.loc_locks[location] = Lock()  


            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data[location] if location in self.sensor_data  \
            else None



    def set_data(self, location, data):

        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):

        self.thread.join()


class Worker(Thread):
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device
        self.conditon = Condition()  
        self.working = True
        self.scripts = []
        self.work_cond = Condition()  
        self.work_lock = Lock()  
        self.is_working = False

    def set_scripts(self, scripts):  
        self.scripts = scripts

        self.work_lock.acquire()
        self.is_working = True

        self.conditon.acquire()

        self.work_lock.release()

        self.conditon.notify_all()
        self.conditon.release()

    def kill(self):  
        self.working = False
        self.conditon.acquire()
        self.conditon.notify()
        self.conditon.release()

    def run(self):

        while True:
            self.conditon.acquire()
            self.conditon.wait()
            self.conditon.release()

            if not self.working:
                break

            for (script, location) in self.scripts:

                self.device.loc_locks[location].acquire()  

                script_data = []
                
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                

                data = self.device.get_data(location)

                if data is not None:
                    script_data.append(data)

                if script_data:
                    
                    result = script.run(script_data)
                    
                    
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                self.device.loc_locks[location].release()  

            
            self.work_lock.acquire()
            self.is_working = False

            self.work_cond.acquire()
            self.work_cond.notify()
            self.work_cond.release()

            self.work_lock.release()


class DeviceThread(Thread):

    def __init__(self, device):

        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
    @staticmethod
    def chunkify(lst, size):  
        return [lst[i::size] for i in range(size)]

    def run(self):
        
        
        
        list_of_workers = []
        for i in range(8):
            list_of_workers.append(Worker(self.device))
        
        for worker in list_of_workers:
            worker.start()

        while True:
            
            self.device.neighbours = self.device.supervisor.get_neighbours()

            
            if self.device.neighbours is None:
                for worker in list_of_workers:
                    worker.kill()
                    if worker.isAlive():
                        worker.join()
                break
            
            self.device.timepoint_done.wait()

            
            list_of_scripts = self.chunkify(self.device.scripts, 8)
            for i in range(8):
                list_of_workers[i].set_scripts(list_of_scripts[i])

            for worker in list_of_workers:  
                worker.work_lock.acquire()
                if worker.is_working:
                    worker.work_cond.acquire()
                    worker.work_lock.release()

                    worker.work_cond.wait()
                    worker.work_cond.release()
                else:
                    worker.work_lock.release()

            self.device.timepoint_done.clear()
            self.device.barrier.wait()          
